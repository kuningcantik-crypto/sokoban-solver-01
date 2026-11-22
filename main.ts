// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Strategy (built for this competition's generator, not a copy of the starter):
//   1. Split the level into rooms. The generator lays a hallway across the board
//      with 1-4 rooms off it, each room holding 3-4 boxes and as many goals, and a
//      box pushed out of its room can never come back, so the level's push-optimal
//      count P* is the sum of the rooms' optima. Solving each room on its own is
//      therefore push-optimal for the whole level, and each room is tiny (a 6x6
//      slot), where a strong search is cheap.
//   2. Solve each room with A* over PUSHES (not moves: only pushes are scored).
//      A state is the sorted box squares plus the player's reachable region,
//      named by its lowest square. The heuristic is the minimum-cost matching
//      between boxes and goals over precomputed grid distances (each push moves
//      one box one square, so the matched distance can drop by at most 1 per
//      push: admissible, hence the first goal popped from the heap is optimal).
//      Deadlock pruning: dead squares (a box that can never be pushed to any
//      goal, computed by pulling boxes back from the goals), frozen 2x2 blocks
//      of walls and boxes, and corner squares.
//   3. Fallbacks, all deterministic (work budgets, never the clock, so two runs
//      write identical bytes): if A* exhausts its expansion budget on a room,
//      retry that room with breadth-first push search; if the level does not
//      split into rooms, run push BFS on the whole level; if that also fails,
//      run a move-based A* with the same deadlock pruning, which still solves
//      the level (with more pushes than P*, worth partial credit) instead of
//      giving up. An unsolved level writes an empty solution, which is valid.
//   4. Reconstruct the player's moves by replaying the push sequence and
//      walking the player to each push square with BFS (deterministic order).
//
// Node 22, TypeScript with erasable syntax only, standard library only.
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expansions / states), never the wall clock: deterministic.
const ASTAR_EXPANSIONS_PER_ROOM = 150_000;
const BFS_STATES_PER_ROOM = 250_000;
const BFS_STATES_WHOLE_LEVEL = 250_000;
const MOVE_ASTAR_EXPANSIONS = 200_000;

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pushed onto some goal (pull back from the goals).
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  while (queue.length > 0) {
    const x = queue.shift()!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y >= 0 && y < wall.length && !wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Squares the player reaches from `from` with `occupied` boxes in the way; marks `seen` with `stamp`, returns lowest.
function reach(level: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q >= 0 && q < wall.length && wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal.
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Corner deadlock: a box on a non-goal square with walls on two perpendicular sides can never move.
function cornerDead(level: Level, goal: Uint8Array, cell: number): boolean {
  if (goal[cell]) return false;
  const { w, wall } = level;
  const up = wall[cell - w];
  const down = wall[cell + w];
  const left = wall[cell - 1];
  const right = wall[cell + 1];
  return (up && left) || (up && right) || (down && left) || (down && right);
}

// All-pairs grid distances (ignoring boxes) from each goal: dist[goalIndex][cell].
function goalDistances(level: Level): number[][] {
  const { w, wall, goals } = level;
  const size = wall.length;
  const out: number[][] = [];
  for (const g of goals) {
    const dist = new Array<number>(size).fill(Infinity);
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (q >= 0 && q < size && !wall[q] && dist[q] === Infinity) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

// Minimum-cost matching between boxes and goals (DP over goal subsets); Infinity when unmatchable.
// Admissible push heuristic: each push moves one box one square, so the matched distance drops by at most 1.
function matchingHeuristic(boxes: number[], dist: number[][], nGoals: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    let sum = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < nGoals; g++) if (dist[g]![b]! < m) m = dist[g]![b]!;
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  }
  const full = (1 << nGoals) - 1;
  const dp = new Array<number>(1 << nGoals).fill(Infinity);
  dp[0] = 0;
  for (let mask = 0; mask <= full; mask++) {
    const i = popcount(mask);
    if (i >= n || dp[mask] === Infinity) continue;
    for (let g = 0; g < nGoals; g++) {
      if (mask & (1 << g)) continue;
      const d = dist[g]![boxes[i]!]!;
      if (d === Infinity) continue;
      const nm = mask | (1 << g);
      const v = dp[mask]! + d;
      if (v < dp[nm]!) dp[nm] = v;
    }
  }
  let best = Infinity;
  for (let mask = 0; mask <= full; mask++) {
    if (popcount(mask) === n && dp[mask]! < best) best = dp[mask]!;
  }
  return best;
}

function popcount(x: number): number {
  let c = 0;
  while (x > 0) {
    c += x & 1;
    x >>= 1;
  }
  return c;
}

// Shortest walk from `from` to `to` around walls and boxes, lower LURD letters (u,d,l,r order on ties).
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

type PushSolution = { pushes: number; moves: string };

// Rebuild the move string from a found state: walk the push chain back, then replay it forward.
function reconstruct(
  level: Level,
  statesBoxes: number[][],
  parent: number[],
  fromCell: number[],
  toCell: number[],
  found: number,
): string {
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  const occupied = new Uint8Array(level.wall.length);
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = fromCell[s]!;
    const to = toCell[s]!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * level.w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

// A* over pushes. Returns the push-optimal solution, or null when the expansion budget runs out.
function solvePushAStar(level: Level, maxExpansions: number): PushSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = goalDistances(level);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b] || cornerDead(level, goal, b))) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const fromCell: number[] = [-1];
  const toCell: number[] = [-1];
  const best = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = reach(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  best.set(keyOf(level.boxes, startLow), 0);

  // Heap of [f, g, state], ties go to larger g (deeper) first, then earlier state.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = matchingHeuristic(level.boxes, dist, goals.length);
  if (h0 === Infinity) return null;
  heapPush([h0, 0, 0]);
  let expanded = 0;
  while (heap.length > 0) {
    const [, g, s] = heapPop();
    if (sg[s] !== g) continue;
    const boxes = statesBoxes[s]!;
    if (solved(boxes)) {
      return { pushes: g, moves: reconstruct(level, statesBoxes, parent, fromCell, toCell, s) };
    }
    if (expanded >= maxExpansions) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t] || cornerDead(level, goal, t)) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        let low = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          low = reach(level, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const ng = g + 1;
        const prevIdx = best.get(key);
        if (prevIdx !== undefined && sg[prevIdx]! <= ng) continue;
        const h = matchingHeuristic(next, dist, goals.length);
        if (h === Infinity) continue;
        const idx = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        sg.push(ng);
        parent.push(s);
        fromCell.push(b);
        toCell.push(t);
        best.set(key, idx);
        heapPush([ng + h, ng, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

// Breadth-first push search (the published method in the kit), used as a fallback
// when A* exhausts its budget: same pruning, states expanded in push order, so the
// first solved state found is push-optimal.
function solvePushBFS(level: Level, maxStates: number): PushSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b] || cornerDead(level, goal, b))) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const fromCell: number[] = [-1];
  const toCell: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  index.set(keyOf(level.boxes, reach(level, occupied, level.player, seen, ++stamp, stack), size), 0);
  for (const b of level.boxes) occupied[b] = 0;
  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statesPlayer[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t] || cornerDead(level, goal, t)) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(level, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || index.has(key)) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(head);
        fromCell.push(b);
        toCell.push(t);
        if (solved(next)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  let pushes = 0;
  for (let s = found; s > 0; s = parent[s]!) pushes++;
  return { pushes, moves: reconstruct(level, statesBoxes, parent, fromCell, toCell, found) };
}

// Move-based A* (steps cost 1, like the house baseline) with the same deadlock
// pruning. It finds the fewest MOVES, not the fewest pushes, so it is only a
// last-resort fallback: a solved level with extra pushes still scores.
function solveMoveAStar(level: Level, maxExpansions: number): string {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = goalDistances(level);
  const est = (bs: number[]): number => {
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (let gi = 0; gi < goals.length; gi++) if (dist[gi]![b]! < m) m = dist[gi]![b]!;
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };
  const key = (p: number, bs: number[]): string => `${p}|${bs.join(',')}`;
  const sp: number[] = [level.player];
  const sb: number[][] = [level.boxes];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(level.player, level.boxes), 0]]);
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  const h0 = est(level.boxes);
  if (h0 === Infinity) return '';
  heapPush([h0, 0, 0]);
  let expanded = 0;
  while (heap.length > 0 && expanded < maxExpansions) {
    const [, g, s] = heapPop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goal[b])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t]! + moves;
      return moves;
    }
    expanded++;
    const p = sp[s]!;
    for (const [dx, dy, letter] of DIRS) {
      const d = dy * w + dx;
      const q = p + d;
      if (q < 0 || q >= size || wall[q]) continue;
      let nb = bs;
      let move = letter;
      const at = bs.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || bs.includes(q + d)) continue;
        if (!live[q + d] || cornerDead(level, goal, q + d)) continue;
        nb = bs.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
        move = letter.toUpperCase();
      }
      const k = key(q, nb);
      const seenIdx = best.get(k);
      if (seenIdx !== undefined && sg[seenIdx]! <= g + 1) continue;
      const h = est(nb);
      if (h === Infinity) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(move);
      best.set(k, t);
      heapPush([g + 1 + h, g + 1, t]);
    }
  }
  return '';
}

type Room = { squares: number[]; boxes: number[]; goals: number[]; done?: boolean };

// Split the level into rooms by cutting the dead squares of the player's row
// (the hallway: a box there reaches no goal). Null when the split is invalid.
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q >= 0 && q < wall.length && !wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve room by room: other rooms' boxes stand as walls (on their goals once done).
function solveByRooms(level: Level): PushSolution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = solvePushAStar(sub, ASTAR_EXPANSIONS_PER_ROOM);
    if (solved === null) solved = solvePushBFS(sub, BFS_STATES_PER_ROOM);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

function solve(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.every((b) => level.goals.includes(b))) return '';
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  const wholeAStar = solvePushAStar(level, ASTAR_EXPANSIONS_PER_ROOM);
  if (wholeAStar !== null) return wholeAStar.moves;
  const wholeBFS = solvePushBFS(level, BFS_STATES_WHOLE_LEVEL);
  if (wholeBFS !== null) return wholeBFS.moves;
  return solveMoveAStar(level, MOVE_ASTAR_EXPANSIONS);
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
