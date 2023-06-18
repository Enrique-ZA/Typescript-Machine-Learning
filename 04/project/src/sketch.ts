// using p5js - https://p5js.org/download/

const colorWhite: [number, number, number, number] = [255, 255, 255, 255];
const colorBlack: [number, number, number, number] = [0, 0, 0, 255];

function preload(): void {
    const m: Matrix = matrixCreate(2, 2);
    const n: Matrix = matrixCreate(2, 2);
    matrixFill(m, 1);
    matrixFill(n, 1);
    matrixSum(m, n);
    matrixPrint(m);
}

function setup(): void {
    // @ts-ignore
    createCanvas(600, 600);    
}

function draw(){
    update();

    // @ts-ignore
    background(colorBlack);

    drawText(colorWhite, 20, 'word', 10, 30);
}

function update(){
}

interface Matrix {
    rows: number;
    cols: number;
    stride?: number;
    data: number[][];
}

function matrixCreate(numRows: number, numCols: number): Matrix {
    let mat: number[][] = [];
    for(let i = 0; i < numRows; ++i){
        mat[i] = [];
        for(let j = 0; j < numCols; ++j){
            mat[i][j] = 0;
        }
    }
    return {rows: numRows, cols: numCols, data: mat};
}

function matrixMult(dst: Matrix, a: Matrix, b: Matrix): void {
    if(a.cols !== b.rows){
       console.error('matrix dot error'); 
    }
    const n = a.cols;
    if(a.rows !== dst.rows || a.cols !== dst.cols){
       console.error('matrix dot error'); 
    }
    for(let i = 0; i < dst.rows; ++i){
        for(let j = 0; j < dst.cols; ++j){
            dst.data[i][j] = 0;
            for(let k = 0; k < n; ++k){
                dst.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
}

function matrixFill(mat: Matrix, num: number): void {
    for(let i = 0; i < mat.rows; ++i){
        for(let j = 0; j < mat.cols; ++j){
            mat.data[i][j] = num;
        }
    }
}

function matrixSum(dst: Matrix, a: Matrix): void {
    if (dst.rows !== a.rows || dst.cols !== a.cols) {
        return;
    }
    for(let i = 0; i < dst.rows; ++i){
        for(let j = 0; j < dst.cols; ++j){
            dst.data[i][j] += a.data[i][j];
        }
    }
}

function matrixRandomize(mat: Matrix, low: number, high: number): void {
    const seed: number[] = cyrb128(Math.random().toString());
    const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);
    for(let i = 0; i < mat.rows; ++i){
        for(let j = 0; j < mat.cols; ++j){
            mat.data[i][j] = rnd() * (high - low) + low;
        }
    }
}

function matrixPrint(mat: Matrix): void {
    console.table(mat.data);
}

class NN {
    constructor() {
    }
}

function sfc32(a: number, b: number, c: number, d: number) {
    return function() {
        a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0; 
        var t = (a + b) | 0;
        a = b ^ b >>> 9;
        b = c + (c << 3) | 0;
        c = (c << 21 | c >>> 11);
        d = d + 1 | 0;
        t = t + d | 0;
        c = c + t | 0;
        return (t >>> 0) / 4294967296;
    }
}

function cyrb128(str: string): number[] {
    let h1 = 1779033703, h2 = 3144134277,
        h3 = 1013904242, h4 = 2773480762;
    for (let i = 0, k; i < str.length; i++) {
        k = str.charCodeAt(i);
        h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
        h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
        h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
        h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
    }
    h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
    h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
    h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
    h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
    return [(h1^h2^h3^h4)>>>0, (h2^h1)>>>0, (h3^h1)>>>0, (h4^h1)>>>0];
}

function drawText(color: [number, number, number, number], txtSize: number, txt: string, posX: number, posY: number): void {
    // @ts-ignore
    fill(color);
    // @ts-ignore
    textSize(txtSize);
    // @ts-ignore
    text(txt, posX, posY);
}
