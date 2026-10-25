"""Shortcut-aware training: can a network be made to ignore the environment?

Four pre-specified methods, all on the frozen CNN body (Conv1D-64, BN, Dropout,
GAP, Dense-8) and all selected by *within-environment* validation AUC:

* ``select``      class-balanced BCE as in the published ensemble; only the
                  model-selection rule changes (control for the new selection).
* ``balanced``    every (environment, class) cell of the training set gets equal
                  total weight, so the environment carries no label information;
                  environments with a single class in training get weight 0.
* ``adversarial`` an environment head behind a gradient-reversal layer
                  (DANN; lambda ramped 0 -> 1) pushes the body to discard
                  environment information.
* ``ranking``     pairwise logistic loss on (elevated, non-elevated) pairs drawn
                  from the *same* environment only: the objective is the
                  within-environment AUC itself.

Early stopping and the reported member use the best validation
within-environment AUC (patience 12, at most ``epochs``); if no validation
environment contains both classes, pooled validation AUC is used instead.
"""

from __future__ import annotations

import numpy as np

from analysis.revision_common import WeightedAUC

METHODS = ("select", "balanced", "adversarial", "ranking")


def _selection_score(y, s, env):
    within = WeightedAUC(y, s, env)()
    return within if np.isfinite(within) else WeightedAUC(y, s)()


def _build(tf, n_features, n_env, adversarial):
    layers = tf.keras.layers

    @tf.custom_gradient
    def reverse(x, lam):
        def grad(dy):
            return -lam * dy, tf.zeros_like(lam)
        return tf.identity(x), grad

    class GradientReversal(layers.Layer):
        def call(self, x, lam):
            return reverse(x, tf.reshape(lam, [-1, 1]))

    inp = layers.Input(shape=(30, n_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    feat = layers.Dense(8, activation="relu")(x)
    logit = layers.Dense(1)(feat)
    if not adversarial:
        return tf.keras.Model(inp, logit), None
    lam_in = layers.Input(shape=(), name="lam")
    r = GradientReversal()(feat, lam_in)
    r = layers.Dense(16, activation="relu")(r)
    env_logits = layers.Dense(n_env)(r)
    return tf.keras.Model(inp, logit), tf.keras.Model([inp, lam_in], [logit, env_logits])


def train_one(tf, method, x_tr, y_tr, env_tr, x_va, y_va, env_va, seed, epochs=100,
              batch_size=32, learning_rate=5e-4, patience=12, pairs_per_side=16):
    if method not in METHODS:
        raise ValueError(method)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    levels = np.unique(env_tr)
    env_idx = np.searchsorted(levels, env_tr)
    scorer, adv_model = _build(tf, x_tr.shape[-1], len(levels), method == "adversarial")
    trainable = (adv_model or scorer).trainable_variables
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # per-window weights
    counts = np.bincount(y_tr, minlength=2).astype(float)
    counts[counts == 0] = 1
    w_class = np.where(y_tr == 1, len(y_tr) / (2 * counts[1]), len(y_tr) / (2 * counts[0]))
    if method == "balanced":
        w = np.zeros(len(y_tr))
        cells = {(e, c): np.flatnonzero((env_idx == e) & (y_tr == c)) for e in range(len(levels)) for c in (0, 1)}
        usable = [e for e in range(len(levels)) if len(cells[(e, 0)]) and len(cells[(e, 1)])]
        for e in usable:
            for c in (0, 1):
                w[cells[(e, c)]] = 1.0 / len(cells[(e, c)])
        w = w * (w > 0).sum() / w.sum()
    else:
        w = w_class
    # ranking: environments with both classes, sampled in proportion to their pair count
    pos_by_env = {e: np.flatnonzero((env_idx == e) & (y_tr == 1)) for e in range(len(levels))}
    neg_by_env = {e: np.flatnonzero((env_idx == e) & (y_tr == 0)) for e in range(len(levels))}
    rank_envs = [e for e in range(len(levels)) if len(pos_by_env[e]) and len(neg_by_env[e])]
    rank_p = np.array([len(pos_by_env[e]) * len(neg_by_env[e]) for e in rank_envs], float)
    rank_p = rank_p / rank_p.sum() if len(rank_envs) else rank_p

    x_tr_t = tf.constant(x_tr)

    @tf.function
    def step_weighted(xb, yb, wb):
        with tf.GradientTape() as tape:
            logit = scorer(xb, training=True)[:, 0]
            loss = tf.reduce_sum(bce(yb[:, None], logit[:, None]) * wb) / tf.reduce_sum(wb)
        opt.apply_gradients(zip(tape.gradient(loss, scorer.trainable_variables), scorer.trainable_variables))
        return loss

    @tf.function
    def step_adversarial(xb, yb, wb, eb, lam):
        with tf.GradientTape() as tape:
            logit, env_logits = adv_model([xb, tf.fill([tf.shape(xb)[0]], lam)], training=True)
            task = tf.reduce_sum(bce(yb[:, None], logit) * wb) / tf.reduce_sum(wb)
            loss = task + sce(eb, env_logits)
        opt.apply_gradients(zip(tape.gradient(loss, trainable), trainable))
        return task

    @tf.function
    def step_ranking(xb, same_env_pair):
        # one forward pass over a batch mixing several environments (so batch
        # normalization sees the same mixture as at inference); the loss only
        # counts (elevated, non-elevated) pairs from the same environment
        with tf.GradientTape() as tape:
            s_all = scorer(xb, training=True)[:, 0]
            diff = s_all[:, None] - s_all[None, :]
            loss = tf.reduce_sum(tf.nn.softplus(-diff) * same_env_pair) / tf.reduce_sum(same_env_pair)
        opt.apply_gradients(zip(tape.gradient(loss, scorer.trainable_variables), scorer.trainable_variables))
        return loss

    def predict(x):
        return tf.sigmoid(scorer.predict(x, batch_size=512, verbose=0)[:, 0]).numpy()

    steps = int(np.ceil(len(y_tr) / batch_size))
    best, best_weights, wait, history = -np.inf, None, 0, []
    for epoch in range(epochs):
        if method == "ranking":
            if not rank_envs:
                break
            per_env = max(2, pairs_per_side // 2)
            for _ in range(steps):
                chosen = rng.choice(len(rank_envs), size=min(4, len(rank_envs)), replace=False, p=rank_p)
                idx, grp, lab = [], [], []
                for g, k in enumerate(chosen):
                    e = rank_envs[k]
                    idx += list(rng.choice(pos_by_env[e], per_env)) + list(rng.choice(neg_by_env[e], per_env))
                    grp += [g] * (2 * per_env)
                    lab += [1] * per_env + [0] * per_env
                grp, lab = np.array(grp), np.array(lab)
                mask = ((grp[:, None] == grp[None, :]) & (lab[:, None] == 1) & (lab[None, :] == 0)).astype("float32")
                step_ranking(tf.gather(x_tr_t, np.array(idx)), tf.constant(mask))
        else:
            order = rng.permutation(len(y_tr))
            if method == "balanced":
                order = order[w[order] > 0]
            lam = 2.0 / (1.0 + np.exp(-10.0 * epoch / max(epochs - 1, 1))) - 1.0
            for k in range(0, len(order), batch_size):
                b = order[k:k + batch_size]
                xb, yb = tf.gather(x_tr_t, b), tf.constant(y_tr[b], tf.float32)
                wb = tf.constant(w[b], tf.float32)
                if method == "adversarial":
                    step_adversarial(xb, yb, wb, tf.constant(env_idx[b]), tf.constant(lam, tf.float32))
                else:
                    step_weighted(xb, yb, wb)
        score = _selection_score(y_va, predict(x_va), env_va)
        history.append(float(score))
        if score > best + 1e-4:
            best, best_weights, wait = score, scorer.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_weights is not None:
        scorer.set_weights(best_weights)
    return predict, {"epochs": len(history), "best_validation_within_env_auc": float(best)}


def model_debiased(tf, method, x_tr, y_tr, x_va, y_va, x_te, meta_tr, meta_va, epochs, members, **_):
    env_tr = meta_tr["condition"].astype(str).to_numpy()
    env_va = meta_va["condition"].astype(str).to_numpy()
    va, te, info = [], [], {"members": [], "method": method}
    for seed in members:
        predict, m = train_one(tf, method, x_tr, y_tr, env_tr, x_va, y_va, env_va, seed, epochs)
        va.append(predict(x_va))
        te.append(predict(x_te))
        info["members"].append({"seed": seed, **m})
    return np.mean(va, axis=0), np.mean(te, axis=0), info
