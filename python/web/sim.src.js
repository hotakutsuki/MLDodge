// Motor + cerebro portados 1:1 desde src/game/engine.py (campeón gen590).
// Funciones PURAS (sin RNG) validadas contra Python en ref_cases.json.
(function (root) {
  const CFG = {
    ARENA_W: 420, ARENA_H: 420, PLAYER_R: 10, ENEMY_R: 9, SPEED: 120,
    SPAWN: 0.6, MAX_LIFE: 35, DESPAWN_M: 72,
    ROWS: 7, COLS: 7, CELL: 60, INFLUENCE: 70, FUTURE_DT: 0.5,
    FIXED_DT: 1 / 128,
  };
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  // (rows*cols*3) aplanado en orden fila-mayor: ((r*COLS)+c)*3 + canal.
  function visionGrid(px, py, enemies) {
    const { ROWS, COLS, CELL, INFLUENCE, FUTURE_DT, SPEED, ARENA_W, ARENA_H } = CFG;
    const cr = ROWS >> 1, cc = COLS >> 1;
    const out = new Float32Array(ROWS * COLS * 3);
    const cxs = new Float32Array(COLS), cys = new Float32Array(ROWS);
    for (let c = 0; c < COLS; c++) cxs[c] = px + (c - cc) * CELL;
    for (let r = 0; r < ROWS; r++) cys[r] = py + (r - cr) * CELL;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const gx = cxs[c], gy = cys[r];
        let danger = 0, closing = 0, fut = 0;
        for (let i = 0; i < enemies.length; i++) {
          const e = enemies[i];
          const d = Math.hypot(gx - e.x, gy - e.y);
          const prox = clamp01(1 - d / INFLUENCE);
          if (prox > danger) {
            danger = prox;
            const ddx = e.x - px, ddy = e.y - py;
            const dist = Math.hypot(ddx, ddy);
            closing = dist <= 1e-6 ? 0
              : Math.max(0, -(e.vx * ddx + e.vy * ddy) / dist / SPEED);
          }
          const fx = e.x + e.vx * FUTURE_DT, fy = e.y + e.vy * FUTURE_DT;
          const fprox = clamp01(1 - Math.hypot(gx - fx, gy - fy) / INFLUENCE);
          if (fprox > fut) fut = fprox;
        }
        if (gx < 0 || gx > ARENA_W || gy < 0 || gy > ARENA_H) { danger = 1; closing = 0; }
        const base = (r * COLS + c) * 3;
        out[base] = danger; out[base + 1] = closing; out[base + 2] = fut;
      }
    }
    return out;
  }

  function encodeObservation(px, py, enemies) {
    const { ARENA_W: w, ARENA_H: h, PLAYER_R: pr } = CFG;
    const obs = new Float32Array(153);
    obs[0] = (px - w / 2) / (w / 2);
    obs[1] = (py - h / 2) / (h / 2);
    obs[2] = (px - pr) / w;
    obs[3] = (w - px - pr) / w;
    obs[4] = (py - pr) / h;
    obs[5] = (h - py - pr) / h;
    const g = visionGrid(px, py, enemies);
    obs.set(g, 6);
    return obs;
  }

  // MLP: relu en ocultas, lineal en la última, argmax. brain.layers[i] = {W:[out][in], b:[out]}.
  function decideAction(brain, obs) {
    let a = obs;
    const L = brain.layers.length;
    for (let li = 0; li < L; li++) {
      const { W, b } = brain.layers[li];
      const out = new Float32Array(W.length);
      for (let o = 0; o < W.length; o++) {
        const row = W[o];
        let s = b[o];
        for (let j = 0; j < row.length; j++) s += row[j] * a[j];
        if (li < L - 1 && s < 0) s = 0; // relu salvo la última
        out[o] = s;
      }
      a = out;
    }
    let best = 0;
    for (let o = 1; o < a.length; o++) if (a[o] > a[best]) best = o;
    return best;
  }

  // --- Motor con RNG propio (mulberry32) para el juego en el navegador ---
  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6D2B79F5;
      let x = t;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function newGame(seed) {
    return {
      px: CFG.ARENA_W / 2, py: CFG.ARENA_H / 2, enemies: [],
      t: 0, spawnTimer: 0, over: false, reason: "", rng: mulberry32(seed >>> 0),
    };
  }

  function spawnEnemy(st) {
    const { ENEMY_R, ARENA_W, ARENA_H, SPEED } = CFG;
    const side = Math.floor(st.rng() * 4) % 4;
    const m = ENEMY_R + 2;
    let ex, ey;
    if (side === 0) { ex = m + st.rng() * (ARENA_W - 2 * m); ey = -m; }
    else if (side === 1) { ex = m + st.rng() * (ARENA_W - 2 * m); ey = ARENA_H + m; }
    else if (side === 2) { ex = -m; ey = m + st.rng() * (ARENA_H - 2 * m); }
    else { ex = ARENA_W + m; ey = m + st.rng() * (ARENA_H - 2 * m); }
    const dx = st.px - ex, dy = st.py - ey;
    const dist = Math.hypot(dx, dy) + 1e-6;
    st.enemies.push({ x: ex, y: ey, vx: (dx / dist) * SPEED, vy: (dy / dist) * SPEED, age: 0 });
  }

  function stepGame(st, action, dt) {
    if (st.over || dt <= 0) return;
    const C = CFG;
    let dx = 0, dy = 0;
    if (action === 0) dy = -C.SPEED;
    else if (action === 1) dy = C.SPEED;
    else if (action === 2) dx = -C.SPEED;
    else if (action === 3) dx = C.SPEED;
    st.px += dx * dt; st.py += dy * dt;
    if (st.px - C.PLAYER_R < 0 || st.px + C.PLAYER_R > C.ARENA_W ||
        st.py - C.PLAYER_R < 0 || st.py + C.PLAYER_R > C.ARENA_H) {
      st.over = true; st.reason = "boundary"; return;
    }
    const touch = C.PLAYER_R + C.ENEMY_R, touchSq = touch * touch;
    const survivors = [];
    for (let i = 0; i < st.enemies.length; i++) {
      const e = st.enemies[i];
      e.age += dt; e.x += e.vx * dt; e.y += e.vy * dt;
      const ddx = e.x - st.px, ddy = e.y - st.py;
      if (ddx * ddx + ddy * ddy <= touchSq) { st.over = true; st.reason = "enemy"; return; }
      if (e.age >= C.MAX_LIFE ||
          e.x < -C.DESPAWN_M || e.x > C.ARENA_W + C.DESPAWN_M ||
          e.y < -C.DESPAWN_M || e.y > C.ARENA_H + C.DESPAWN_M) continue;
      survivors.push(e);
    }
    st.enemies = survivors;
    st.t += dt; st.spawnTimer += dt;
    while (st.spawnTimer >= C.SPAWN) { st.spawnTimer -= C.SPAWN; spawnEnemy(st); }
  }

  root.SIM = { CFG, visionGrid, encodeObservation, decideAction, newGame, stepGame };
})(typeof module !== "undefined" ? module.exports : window);
