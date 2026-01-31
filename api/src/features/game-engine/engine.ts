import type {
    Board,
    PlayerColor,
    GameState,
    Move,
    MoveResult,
    PlayerState,
    GameSummary,
} from './types';
import {
    PLAYER_ORDER,
    createEmptyBoard,
    createPlayerState,
    getNextPlayer,
} from './types';
import { ALL_PIECE_IDS, PIECE_BY_ID } from '../pieces/service';
import { validateMove, generateLegalMoves, getPieceCells } from './validation';

export class BlokusGame {
    private state: GameState;
    private moveHistory: Move[] = [];

    constructor() {
        this.state = this.createInitialState();
    }

    private createInitialState(): GameState {
        const players: Record<PlayerColor, PlayerState> = {
            blue: createPlayerState('blue', ALL_PIECE_IDS),
            yellow: createPlayerState('yellow', ALL_PIECE_IDS),
            red: createPlayerState('red', ALL_PIECE_IDS),
            green: createPlayerState('green', ALL_PIECE_IDS),
        };

        return {
            board: createEmptyBoard(),
            players,
            currentPlayer: 'blue',
            turnNumber: 1,
            isGameOver: false,
            consecutivePasses: 0,
        };
    }

    // Get current game state (read-only copy)
    getState(): Readonly<GameState> {
        return this.state;
    }

    // Get move history
    getMoveHistory(): readonly Move[] {
        return this.moveHistory;
    }

    // Get current player
    getCurrentPlayer(): PlayerColor {
        return this.state.currentPlayer;
    }

    // Check if game is over
    isGameOver(): boolean {
        return this.state.isGameOver;
    }

    // Get all legal moves for current player
    getLegalMoves(): Move[] {
        if (this.state.isGameOver) {
            return [];
        }

        const playerState = this.state.players[this.state.currentPlayer];
        if (playerState.isEliminated) {
            return [];
        }

        return generateLegalMoves(this.state.board, playerState);
    }

    // Check if current player has any legal moves
    hasLegalMoves(): boolean {
        return this.getLegalMoves().length > 0;
    }

    // Apply a move to the game
    makeMove(move: Move): MoveResult {
        if (this.state.isGameOver) {
            return { success: false, error: 'Game is already over' };
        }

        if (move.player !== this.state.currentPlayer) {
            return { success: false, error: `Not ${move.player}'s turn` };
        }

        const playerState = this.state.players[move.player];
        const validation = validateMove(this.state.board, playerState, move);

        if (!validation.valid) {
            return { success: false, error: validation.error };
        }

        // Apply the move
        const cells = getPieceCells(move)!;
        const piece = PIECE_BY_ID.get(move.pieceId)!;

        // Update board
        for (const { row, col } of cells) {
            this.state.board[row][col] = move.player;
        }

        // Update player state
        playerState.remainingPieces.delete(move.pieceId);
        playerState.hasPlayed = true;
        playerState.score += piece.size;

        // Record move
        this.moveHistory.push(move);

        // Reset consecutive passes
        this.state.consecutivePasses = 0;

        // Advance to next player
        this.advanceTurn();

        return { success: true, newState: this.state };
    }

    // Pass turn (when player has no legal moves)
    pass(): MoveResult {
        if (this.state.isGameOver) {
            return { success: false, error: 'Game is already over' };
        }

        const playerState = this.state.players[this.state.currentPlayer];

        // Can only pass if no legal moves available
        if (this.hasLegalMoves()) {
            return { success: false, error: 'Cannot pass when legal moves are available' };
        }

        // Mark player as eliminated (no more moves this game)
        playerState.isEliminated = true;
        this.state.consecutivePasses++;

        // Advance to next player
        this.advanceTurn();

        return { success: true, newState: this.state };
    }

    private advanceTurn(): void {
        // Find next non-eliminated player
        let nextPlayer = getNextPlayer(this.state.currentPlayer);
        let checked = 0;

        while (checked < 4) {
            if (!this.state.players[nextPlayer].isEliminated) {
                break;
            }
            nextPlayer = getNextPlayer(nextPlayer);
            checked++;
        }

        // Check if all players are eliminated
        if (checked >= 4 || this.state.consecutivePasses >= 4) {
            this.state.isGameOver = true;
            return;
        }

        this.state.currentPlayer = nextPlayer;
        this.state.turnNumber++;

        // Check if new current player has any moves
        // If not, they need to pass (handled by caller or auto-pass)
    }

    // Auto-pass for players with no moves (useful for simulation)
    autoPassIfNeeded(): boolean {
        if (this.state.isGameOver) {
            return false;
        }

        if (!this.hasLegalMoves()) {
            this.pass();
            return true;
        }

        return false;
    }

    // Get game summary
    getSummary(): GameSummary {
        const scores: Record<PlayerColor, number> = {
            blue: this.state.players.blue.score,
            yellow: this.state.players.yellow.score,
            red: this.state.players.red.score,
            green: this.state.players.green.score,
        };

        // Bonus: +15 if player used all pieces, +5 more if monomino was last
        for (const player of PLAYER_ORDER) {
            const playerState = this.state.players[player];
            if (playerState.remainingPieces.size === 0) {
                scores[player] += 15;
                // Check if last piece was monomino
                const lastMoveByPlayer = [...this.moveHistory]
                    .reverse()
                    .find(m => m.player === player);
                if (lastMoveByPlayer?.pieceId === '1') {
                    scores[player] += 5;
                }
            } else {
                // Subtract remaining piece sizes
                for (const pieceId of playerState.remainingPieces) {
                    const piece = PIECE_BY_ID.get(pieceId);
                    if (piece) {
                        scores[player] -= piece.size;
                    }
                }
            }
        }

        // Determine winner (highest score)
        let winner: PlayerColor | null = null;
        let highestScore = -Infinity;

        for (const player of PLAYER_ORDER) {
            if (scores[player] > highestScore) {
                highestScore = scores[player];
                winner = player;
            } else if (scores[player] === highestScore) {
                winner = null; // Tie
            }
        }

        return {
            winner,
            scores,
            totalMoves: this.moveHistory.length,
            finalBoard: this.state.board,
        };
    }

    // Reset game to initial state
    reset(): void {
        this.state = this.createInitialState();
        this.moveHistory = [];
    }

    // Clone the game (for simulation/AI)
    clone(): BlokusGame {
        const cloned = new BlokusGame();

        // Deep copy state
        cloned.state = {
            board: this.state.board.map(row => [...row]),
            players: {
                blue: { ...this.state.players.blue, remainingPieces: new Set(this.state.players.blue.remainingPieces) },
                yellow: { ...this.state.players.yellow, remainingPieces: new Set(this.state.players.yellow.remainingPieces) },
                red: { ...this.state.players.red, remainingPieces: new Set(this.state.players.red.remainingPieces) },
                green: { ...this.state.players.green, remainingPieces: new Set(this.state.players.green.remainingPieces) },
            },
            currentPlayer: this.state.currentPlayer,
            turnNumber: this.state.turnNumber,
            isGameOver: this.state.isGameOver,
            consecutivePasses: this.state.consecutivePasses,
        };

        cloned.moveHistory = [...this.moveHistory];

        return cloned;
    }

    // Create game from a board state (useful for analyzing images)
    static fromBoard(board: Board, currentPlayer: PlayerColor = 'blue'): BlokusGame {
        const game = new BlokusGame();
        game.state.board = board.map(row => [...row]);
        game.state.currentPlayer = currentPlayer;

        // Analyze board to determine which pieces have been played
        // This is a simplified version - doesn't reconstruct exact move history
        // but does update remaining pieces and scores

        // Import the analyzeGrid function to detect pieces
        // For now, we'll just copy the board and let the caller set up player states

        return game;
    }
}

// Utility: Play a random game (for testing)
export const playRandomGame = (): GameSummary => {
    const game = new BlokusGame();

    while (!game.isGameOver()) {
        const moves = game.getLegalMoves();

        if (moves.length === 0) {
            game.pass();
        } else {
            // Pick random move
            const randomMove = moves[Math.floor(Math.random() * moves.length)];
            game.makeMove(randomMove);
        }
    }

    return game.getSummary();
};
