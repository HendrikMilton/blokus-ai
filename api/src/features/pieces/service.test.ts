import { expect } from 'chai';
import {
    normalizeShape,
    rotateShape,
    flipShape,
    shapeToString,
    findConnectedComponents,
    matchPiece,
    analyzeGrid,
    PIECE_CATALOG,
    ALL_PIECE_IDS,
    type Coord,
    type PieceShape,
} from './service';
import type { GridLabels } from '@blokus-ai/shared';

// Helper to create empty 20x20 grid
const createEmptyGrid = (): GridLabels =>
    Array.from({ length: 20 }, () => Array(20).fill(null));

// Helper to place a piece on a grid
const placePiece = (grid: GridLabels, coords: Coord[], color: 'blue' | 'yellow' | 'red' | 'green') => {
    for (const [r, c] of coords) {
        grid[r][c] = color;
    }
};

describe('pieces service', () => {
    describe('normalizeShape', () => {
        it('should translate shape to origin', () => {
            const shape: PieceShape = [[5, 5], [5, 6], [6, 5]];
            const normalized = normalizeShape(shape);
            expect(normalized).to.deep.equal([[0, 0], [0, 1], [1, 0]]);
        });

        it('should sort coordinates consistently', () => {
            const shape: PieceShape = [[1, 0], [0, 1], [0, 0]];
            const normalized = normalizeShape(shape);
            expect(normalized).to.deep.equal([[0, 0], [0, 1], [1, 0]]);
        });

        it('should handle single cell', () => {
            const shape: PieceShape = [[10, 15]];
            const normalized = normalizeShape(shape);
            expect(normalized).to.deep.equal([[0, 0]]);
        });

        it('should handle empty shape', () => {
            const normalized = normalizeShape([]);
            expect(normalized).to.deep.equal([]);
        });
    });

    describe('rotateShape', () => {
        it('should rotate I3 (horizontal line) to vertical', () => {
            const horizontal: PieceShape = [[0, 0], [0, 1], [0, 2]];
            const rotated = rotateShape(horizontal);
            // After rotation and normalization, should be vertical
            expect(rotated).to.deep.equal([[0, 0], [1, 0], [2, 0]]);
        });

        it('should rotate L shape correctly', () => {
            // L shape: ##
            //          #
            const lShape: PieceShape = [[0, 0], [0, 1], [1, 0]];
            const rotated = rotateShape(lShape);
            // After 90° clockwise rotation
            expect(rotated.length).to.equal(3);
        });
    });

    describe('flipShape', () => {
        it('should flip L shape horizontally', () => {
            // L shape pointing right -> L shape pointing left
            const lShape: PieceShape = [[0, 0], [1, 0], [2, 0], [2, 1]];
            const flipped = flipShape(lShape);
            expect(flipped.length).to.equal(4);
            // Should be mirrored version
            expect(shapeToString(flipped)).to.not.equal(shapeToString(normalizeShape(lShape)));
        });

        it('should not change symmetric shapes', () => {
            // Vertical line is symmetric
            const vertical: PieceShape = [[0, 0], [1, 0], [2, 0]];
            const flipped = flipShape(vertical);
            expect(shapeToString(flipped)).to.equal(shapeToString(normalizeShape(vertical)));
        });
    });

    describe('generateOrientations', () => {
        it('should generate 1 orientation for X piece (fully symmetric)', () => {
            const xPiece = PIECE_CATALOG.find(p => p.id === 'X');
            expect(xPiece).to.exist;
            expect(xPiece!.orientations.length).to.equal(1);
        });

        it('should generate 2 orientations for I5 piece (line)', () => {
            const i5Piece = PIECE_CATALOG.find(p => p.id === 'I5');
            expect(i5Piece).to.exist;
            expect(i5Piece!.orientations.length).to.equal(2);
        });

        it('should generate 4 orientations for T5 piece', () => {
            const t5Piece = PIECE_CATALOG.find(p => p.id === 'T5');
            expect(t5Piece).to.exist;
            expect(t5Piece!.orientations.length).to.equal(4);
        });

        it('should generate 8 orientations for F piece (asymmetric)', () => {
            const fPiece = PIECE_CATALOG.find(p => p.id === 'F');
            expect(fPiece).to.exist;
            expect(fPiece!.orientations.length).to.equal(8);
        });

        it('should generate 1 orientation for O4 piece (square)', () => {
            const o4Piece = PIECE_CATALOG.find(p => p.id === 'O4');
            expect(o4Piece).to.exist;
            expect(o4Piece!.orientations.length).to.equal(1);
        });
    });

    describe('PIECE_CATALOG', () => {
        it('should have 21 pieces', () => {
            expect(PIECE_CATALOG.length).to.equal(21);
        });

        it('should have correct cell counts', () => {
            const sizeCounts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
            for (const piece of PIECE_CATALOG) {
                sizeCounts[piece.size as keyof typeof sizeCounts]++;
            }
            expect(sizeCounts[1]).to.equal(1);  // 1 monomino
            expect(sizeCounts[2]).to.equal(1);  // 1 domino
            expect(sizeCounts[3]).to.equal(2);  // 2 triominoes
            expect(sizeCounts[4]).to.equal(5);  // 5 tetrominoes
            expect(sizeCounts[5]).to.equal(12); // 12 pentominoes
        });

        it('should have unique piece IDs', () => {
            const ids = PIECE_CATALOG.map(p => p.id);
            const uniqueIds = new Set(ids);
            expect(uniqueIds.size).to.equal(ids.length);
        });
    });

    describe('findConnectedComponents', () => {
        it('should find single piece', () => {
            const grid = createEmptyGrid();
            placePiece(grid, [[0, 0], [0, 1], [0, 2]], 'blue');

            const components = findConnectedComponents(grid, 'blue');
            expect(components.length).to.equal(1);
            expect(components[0].length).to.equal(3);
        });

        it('should find multiple separate pieces of same color', () => {
            const grid = createEmptyGrid();
            // Two separate pieces (not edge-connected)
            placePiece(grid, [[0, 0], [0, 1]], 'blue');
            placePiece(grid, [[5, 5], [5, 6], [5, 7]], 'blue');

            const components = findConnectedComponents(grid, 'blue');
            expect(components.length).to.equal(2);
        });

        it('should not connect diagonally adjacent cells', () => {
            const grid = createEmptyGrid();
            // Two cells touching only at corner (diagonal)
            placePiece(grid, [[0, 0]], 'blue');
            placePiece(grid, [[1, 1]], 'blue');

            const components = findConnectedComponents(grid, 'blue');
            expect(components.length).to.equal(2);
        });

        it('should connect edge-adjacent cells', () => {
            const grid = createEmptyGrid();
            // L-shaped piece connected by edges
            placePiece(grid, [[0, 0], [1, 0], [1, 1]], 'blue');

            const components = findConnectedComponents(grid, 'blue');
            expect(components.length).to.equal(1);
            expect(components[0].length).to.equal(3);
        });

        it('should only find components of specified color', () => {
            const grid = createEmptyGrid();
            placePiece(grid, [[0, 0], [0, 1]], 'blue');
            placePiece(grid, [[5, 5], [5, 6]], 'red');

            const blueComponents = findConnectedComponents(grid, 'blue');
            const redComponents = findConnectedComponents(grid, 'red');

            expect(blueComponents.length).to.equal(1);
            expect(redComponents.length).to.equal(1);
        });

        it('should return empty array for color not on board', () => {
            const grid = createEmptyGrid();
            placePiece(grid, [[0, 0]], 'blue');

            const greenComponents = findConnectedComponents(grid, 'green');
            expect(greenComponents.length).to.equal(0);
        });
    });

    describe('matchPiece', () => {
        it('should match monomino (1 cell)', () => {
            const shape: PieceShape = [[5, 5]];
            expect(matchPiece(shape)).to.equal('1');
        });

        it('should match domino (I2)', () => {
            const shape: PieceShape = [[0, 0], [0, 1]];
            expect(matchPiece(shape)).to.equal('2');
        });

        it('should match I5 horizontal', () => {
            const shape: PieceShape = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]];
            expect(matchPiece(shape)).to.equal('I5');
        });

        it('should match I5 vertical (rotated)', () => {
            const shape: PieceShape = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];
            expect(matchPiece(shape)).to.equal('I5');
        });

        it('should match X piece', () => {
            // X shape:  #
            //          ###
            //           #
            const shape: PieceShape = [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]];
            expect(matchPiece(shape)).to.equal('X');
        });

        it('should match L5 in any orientation', () => {
            // Original L5
            const shape1: PieceShape = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]];
            expect(matchPiece(shape1)).to.equal('L5');

            // Rotated L5
            const shape2: PieceShape = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]];
            expect(matchPiece(shape2)).to.equal('L5');
        });

        it('should return null for invalid shape', () => {
            // 6 cells - no such piece
            const shape: PieceShape = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]];
            expect(matchPiece(shape)).to.be.null;
        });

        it('should return null for non-standard shape', () => {
            // 5 cells but not a valid pentomino (disconnected looking when normalized)
            const shape: PieceShape = [[0, 0], [0, 1], [2, 0], [2, 1], [4, 0]];
            expect(matchPiece(shape)).to.be.null;
        });
    });

    describe('analyzeGrid', () => {
        it('should return all pieces as offBoard for empty grid', () => {
            const grid = createEmptyGrid();
            const result = analyzeGrid(grid);

            expect(result.blue.onBoard).to.deep.equal([]);
            expect(result.blue.offBoard).to.deep.equal(ALL_PIECE_IDS.sort());
            expect(result.yellow.onBoard).to.deep.equal([]);
            expect(result.red.onBoard).to.deep.equal([]);
            expect(result.green.onBoard).to.deep.equal([]);
        });

        it('should detect single piece on board', () => {
            const grid = createEmptyGrid();
            // Place X piece for blue
            placePiece(grid, [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]], 'blue');

            const result = analyzeGrid(grid);

            expect(result.blue.onBoard).to.include('X');
            expect(result.blue.offBoard).to.not.include('X');
            expect(result.blue.onBoard.length).to.equal(1);
            expect(result.blue.offBoard.length).to.equal(20);
        });

        it('should detect multiple pieces of same color', () => {
            const grid = createEmptyGrid();
            // Place monomino and I5 for blue (not touching)
            placePiece(grid, [[0, 0]], 'blue'); // monomino '1'
            placePiece(grid, [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4]], 'blue'); // I5

            const result = analyzeGrid(grid);

            expect(result.blue.onBoard).to.include('1');
            expect(result.blue.onBoard).to.include('I5');
            expect(result.blue.onBoard.length).to.equal(2);
        });

        it('should detect pieces for multiple colors', () => {
            const grid = createEmptyGrid();
            placePiece(grid, [[0, 0]], 'blue');      // monomino
            placePiece(grid, [[0, 5], [0, 6]], 'yellow'); // domino
            placePiece(grid, [[5, 0], [5, 1], [5, 2]], 'red'); // I3
            placePiece(grid, [[10, 0], [10, 1], [11, 0]], 'green'); // V3

            const result = analyzeGrid(grid);

            expect(result.blue.onBoard).to.deep.equal(['1']);
            expect(result.yellow.onBoard).to.deep.equal(['2']);
            expect(result.red.onBoard).to.deep.equal(['I3']);
            expect(result.green.onBoard).to.deep.equal(['V3']);
        });

        it('should handle pieces at board edges', () => {
            const grid = createEmptyGrid();
            // I5 along bottom edge
            placePiece(grid, [[19, 0], [19, 1], [19, 2], [19, 3], [19, 4]], 'blue');

            const result = analyzeGrid(grid);
            expect(result.blue.onBoard).to.include('I5');
        });

        it('should handle pieces in corner', () => {
            const grid = createEmptyGrid();
            // V3 in bottom-right corner
            placePiece(grid, [[18, 19], [19, 18], [19, 19]], 'blue');

            const result = analyzeGrid(grid);
            expect(result.blue.onBoard).to.include('V3');
        });
    });
});
