GrandMaster Chess BOT challenge for BRAINROT 

# Chess Bot: Principal Variation Search Engine

This project implements a chess-playing engine that uses Principal Variation Search (PVS) with alpha-beta pruning, piece-square evaluation, and MVV-LVA move ordering for improved performance.

It is designed to play chess by selecting strong moves within a limited time window, making it suitable for integration into real-time or turn-based platforms.

---

## Features

- Principal Variation Search: An optimization of alpha-beta pruning that efficiently explores the game tree.
- Heuristic Evaluation: Combines material count and positional evaluation using piece-square tables.
- MVV-LVA Move Ordering: Most Valuable Victim - Least Valuable Aggressor heuristic to improve pruning efficiency.
- Time-Constrained Search: Iterative deepening within a fixed time limit (default: 2 seconds).
- Aspiration Windows: Dynamically adjust alpha-beta bounds for deeper search efficiency.

---

## How It Works

1. The engine evaluates the board using both material values and precomputed positional weights (piece-square tables).
2. It orders possible moves using heuristics (captures, checks, promotions).
3. It applies PVS with alpha-beta pruning, searching deeper only when promising moves are found.
4. Iterative deepening and aspiration windows help optimize depth vs. time tradeoffs.

---

## File Overview

- `player.py`: The full implementation of the chess engine.
  - `Player.makeMove()`: Returns the best move given a `chess.Board()` position.
  - Uses positional evaluation and pruning to find high-quality moves.

---

## Notes

- Currently configured to search to a depth of 3 within a 2-second window.
- If no move is found in time, it falls back to a random legal move.

---

