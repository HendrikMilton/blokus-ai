import { expect } from 'chai';
import { BlokusGame, playRandomGame } from './engine';
import type { Move } from './types';

describe('BlokusGame', () => {
    describe('initialization', () => {
        it('should start with empty board', () => {
            const game = new BlokusGame();
            const state = game.getState();

            for (let r = 0; r < 20; r++) {
                for (let c = 0; c < 20; c++) {
                    expect(state.board[r][c]).to.be.null;
                }
            }
        });

        it('should start with blue as first player', () => {
            const game = new BlokusGame();
            expect(game.getCurrentPlayer()).to.equal('blue');
        });

        it('should give each player 21 pieces', () => {
            const game = new BlokusGame();
            const state = game.getState();

            expect(state.players.blue.remainingPieces.size).to.equal(21);
            expect(state.players.yellow.remainingPieces.size).to.equal(21);
            expect(state.players.red.remainingPieces.size).to.equal(21);
            expect(state.players.green.remainingPieces.size).to.equal(21);
        });

        it('should not be game over at start', () => {
            const game = new BlokusGame();
            expect(game.isGameOver()).to.be.false;
        });
    });

    describe('first move validation', () => {
        it('should require first move to cover starting corner', () => {
            const game = new BlokusGame();

            // Try placing monomino away from corner - should fail
            const badMove: Move = {
                player: 'blue',
                pieceId: '1',
                position: { row: 5, col: 5 },
                orientationIndex: 0,
            };

            const result = game.makeMove(badMove);
            expect(result.success).to.be.false;
            expect(result.error).to.include('starting corner');
        });

        it('should accept first move on starting corner', () => {
            const game = new BlokusGame();

            // Blue's starting corner is (0, 0)
            const move: Move = {
                player: 'blue',
                pieceId: '1', // monomino
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.true;
            expect(game.getState().board[0][0]).to.equal('blue');
        });

        it('should accept larger piece covering corner', () => {
            const game = new BlokusGame();

            // I3 horizontal: covers (0,0), (0,1), (0,2)
            const move: Move = {
                player: 'blue',
                pieceId: 'I3',
                position: { row: 0, col: 0 },
                orientationIndex: 0, // horizontal
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.true;
            expect(game.getState().board[0][0]).to.equal('blue');
            expect(game.getState().board[0][1]).to.equal('blue');
            expect(game.getState().board[0][2]).to.equal('blue');
        });
    });

    describe('turn order', () => {
        it('should advance to yellow after blue moves', () => {
            const game = new BlokusGame();

            game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            expect(game.getCurrentPlayer()).to.equal('yellow');
        });

        it('should cycle through all four players', () => {
            const game = new BlokusGame();

            // Blue plays at (0,0)
            game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });
            expect(game.getCurrentPlayer()).to.equal('yellow');

            // Yellow plays at (0,19)
            game.makeMove({
                player: 'yellow',
                pieceId: '1',
                position: { row: 0, col: 19 },
                orientationIndex: 0,
            });
            expect(game.getCurrentPlayer()).to.equal('red');

            // Red plays at (19,19)
            game.makeMove({
                player: 'red',
                pieceId: '1',
                position: { row: 19, col: 19 },
                orientationIndex: 0,
            });
            expect(game.getCurrentPlayer()).to.equal('green');

            // Green plays at (19,0)
            game.makeMove({
                player: 'green',
                pieceId: '1',
                position: { row: 19, col: 0 },
                orientationIndex: 0,
            });
            expect(game.getCurrentPlayer()).to.equal('blue');
        });
    });

    describe('subsequent move validation', () => {
        let game: BlokusGame;

        beforeEach(() => {
            game = new BlokusGame();
            // Blue places I3 at corner: (0,0), (0,1), (0,2)
            game.makeMove({
                player: 'blue',
                pieceId: 'I3',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });
            // Yellow, red, green place their first pieces
            game.makeMove({
                player: 'yellow',
                pieceId: '1',
                position: { row: 0, col: 19 },
                orientationIndex: 0,
            });
            game.makeMove({
                player: 'red',
                pieceId: '1',
                position: { row: 19, col: 19 },
                orientationIndex: 0,
            });
            game.makeMove({
                player: 'green',
                pieceId: '1',
                position: { row: 19, col: 0 },
                orientationIndex: 0,
            });
        });

        it('should reject move not touching own color diagonally', () => {
            // Blue's I3 is at (0,0-2). Try placing away from it
            const move: Move = {
                player: 'blue',
                pieceId: '1',
                position: { row: 10, col: 10 },
                orientationIndex: 0,
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.false;
            expect(result.error).to.include('diagonally');
        });

        it('should reject move touching own color edge-to-edge', () => {
            // Try placing monomino at (0,3) - edge-adjacent to I3
            const move: Move = {
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 3 },
                orientationIndex: 0,
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.false;
            expect(result.error).to.include('edge-to-edge');
        });

        it('should accept move touching own color diagonally', () => {
            // Place at (1,3) - diagonal from (0,2)
            const move: Move = {
                player: 'blue',
                pieceId: '1',
                position: { row: 1, col: 3 },
                orientationIndex: 0,
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.true;
        });
    });

    describe('overlap detection', () => {
        it('should reject move overlapping existing piece', () => {
            const game = new BlokusGame();

            // Blue places monomino at (0,0)
            game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            // Yellow tries to place at same spot
            const move: Move = {
                player: 'yellow',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            };

            const result = game.makeMove(move);
            expect(result.success).to.be.false;
            expect(result.error).to.include('overlap');
        });
    });

    describe('piece tracking', () => {
        it('should remove piece from remaining after playing', () => {
            const game = new BlokusGame();

            game.makeMove({
                player: 'blue',
                pieceId: 'I3',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            const state = game.getState();
            expect(state.players.blue.remainingPieces.has('I3')).to.be.false;
            expect(state.players.blue.remainingPieces.size).to.equal(20);
        });

        it('should reject move with already-played piece', () => {
            const game = new BlokusGame();

            game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            // Skip to blue's next turn
            game.makeMove({ player: 'yellow', pieceId: '1', position: { row: 0, col: 19 }, orientationIndex: 0 });
            game.makeMove({ player: 'red', pieceId: '1', position: { row: 19, col: 19 }, orientationIndex: 0 });
            game.makeMove({ player: 'green', pieceId: '1', position: { row: 19, col: 0 }, orientationIndex: 0 });

            // Blue tries to play monomino again
            const result = game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 1, col: 1 },
                orientationIndex: 0,
            });

            expect(result.success).to.be.false;
            expect(result.error).to.include('already played');
        });
    });

    describe('scoring', () => {
        it('should track score based on cells placed', () => {
            const game = new BlokusGame();

            // Blue plays I5 (5 cells) at corner
            game.makeMove({
                player: 'blue',
                pieceId: 'I5',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            expect(game.getState().players.blue.score).to.equal(5);
        });
    });

    describe('legal move generation', () => {
        it('should generate moves for first turn', () => {
            const game = new BlokusGame();
            const moves = game.getLegalMoves();

            // Should have many legal moves (all pieces × orientations that cover corner)
            expect(moves.length).to.be.greaterThan(0);

            // All moves should be for blue
            for (const move of moves) {
                expect(move.player).to.equal('blue');
            }
        });

        it('should include monomino at corner for first move', () => {
            const game = new BlokusGame();
            const moves = game.getLegalMoves();

            const monominoAtCorner = moves.find(
                m => m.pieceId === '1' && m.position.row === 0 && m.position.col === 0
            );

            expect(monominoAtCorner).to.exist;
        });
    });

    describe('game clone', () => {
        it('should create independent copy', () => {
            const game = new BlokusGame();
            game.makeMove({
                player: 'blue',
                pieceId: '1',
                position: { row: 0, col: 0 },
                orientationIndex: 0,
            });

            const cloned = game.clone();

            // Make move on clone
            cloned.makeMove({
                player: 'yellow',
                pieceId: '1',
                position: { row: 0, col: 19 },
                orientationIndex: 0,
            });

            // Original should be unaffected
            expect(game.getCurrentPlayer()).to.equal('yellow');
            expect(cloned.getCurrentPlayer()).to.equal('red');
        });
    });

    describe('random game simulation', () => {
        it('should complete a random game without errors', () => {
            const summary = playRandomGame();

            expect(summary.totalMoves).to.be.greaterThan(0);
            expect(summary.scores.blue).to.be.a('number');
            expect(summary.scores.yellow).to.be.a('number');
            expect(summary.scores.red).to.be.a('number');
            expect(summary.scores.green).to.be.a('number');
        });

        it('should have valid final board state', () => {
            const summary = playRandomGame();

            // Board should have some pieces
            let cellCount = 0;
            for (let r = 0; r < 20; r++) {
                for (let c = 0; c < 20; c++) {
                    if (summary.finalBoard[r][c] !== null) {
                        cellCount++;
                    }
                }
            }

            expect(cellCount).to.be.greaterThan(0);
        });
    });
});
