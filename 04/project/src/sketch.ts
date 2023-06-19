// using p5js - https://p5js.org/download/

const colorWhite: [number, number, number, number] = [255, 255, 255, 255];
const colorBlack: [number, number, number, number] = [0, 0, 0, 255];

const oWidth: number = 1920;
const oHeight: number = 965;
let scaleX: number = 1;

let xor: Xor;
let xorGradient: Xor;
const td: number[][] = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
];

function preload(): void {

    // const ti: Matrix = matrixCreate(td.length, 2);
    // ti.data = matrixSlice(td, 0, ti.cols);
    // const to: Matrix = matrixCreate(td.length, 1);
    // to.data = matrixSlice(td, 2, to.cols);
    // console.log(ti);
    // console.log(to);
    xor = xorCreate(
        matrixCreate(1, 2), // mA0 

        matrixCreate(2, 2), // w1 
        matrixCreate(1, 2), // b1 
        matrixCreate(1, 2), // a1 

        matrixCreate(2, 1), // w2 
        matrixCreate(1, 1), // b2 
        matrixCreate(1, 1), // a2 
    );

    xorGradient = xorCreate(
        matrixCreate(1, 2), // mA0 

        matrixCreate(2, 2), // w1 
        matrixCreate(1, 2), // b1 
        matrixCreate(1, 2), // a1 

        matrixCreate(2, 1), // w2 
        matrixCreate(1, 1), // b2 
        matrixCreate(1, 1), // a2 
    );

    xorRandomize(xor);
    xorRandomize(xorGradient);

    console.table(xorGradient.weightsLayer1.data);

    const epsilon: number = 1e-1;
    // const rate: number = 1e-1;

    // console.log(`loss = ${loss(xor, ti, to)}`);

    for(let i = 0; i < 1; ++i){
        finiteDiff(xor, xorGradient, epsilon, td);
        // xorLearn(xor, xorGradient, rate);
    }

    console.table(xorGradient.weightsLayer1.data);
    // console.log(`loss = ${loss(xor, ti, to)}`);

    // for(let i = 0; i < 2; ++i){
    //     for(let j = 0; j < 2; ++j){
    //         xor.mA0.data[0][0] = i;
    //         xor.mA0.data[0][1] = j;
    //         xorForward(xor);
    //         const y = xor.mA2.data[0][0];
    //         console.log(`${i} ^ ${j} = ${y}`);
    //     }
    // }
}

function setup(): void {
    // @ts-ignore
    createCanvas(windowWidth, windowHeight); // 1920 x 965    
}

// function draw(){
//     // update
//     const w = 120;
//     const h = 120;
//
//     // @ts-ignore
//     const pos1 = {x: width/8, y: height/4};
//     // @ts-ignore
//     const pos2 = {x: pos1.x, y: height/1.5};
//     // @ts-ignore
//     const pos3 = {x: width/2.5, y: pos1.y};
//     // @ts-ignore
//     const pos4 = {x: pos3.x, y: pos2.y};
//     // @ts-ignore
//     const dX = (pos3.x)-(pos1.x);
//     // @ts-ignore
//     const dY = ((pos2.y)-(pos1.y))/2;
//     // @ts-ignore
//     const pos5 = {x: pos3.x+dX, y: pos1.y+dY};
//
//     // @ts-ignore
//     scaleX = width / oWidth;
//     // @ts-ignore
//     scale(scaleX,scaleX);
//
//     // draw
//     // @ts-ignore
//     background(colorBlack);
//
//     // @ts-ignore
//     drawText(colorWhite, 20, xor.weightsLayer1.data.toString(), 10, 30);
//     // @ts-ignore
//     drawText(colorWhite, 20, xor.biasesLayer1.data.toString(), 10, 60);
//     // @ts-ignore
//     drawText(colorWhite, 20, xor.weightsLayer2.data.toString(), 10, 90);
//     // @ts-ignore
//     drawText(colorWhite, 20, xor.biasesLayer2.data.toString(), 10, 120);
//
//     // @ts-ignore
//     push();
//     // @ts-ignore
//     fill(colorWhite);
//     // @ts-ignore
//     noStroke();
//     // @ts-ignore
//     ellipse(pos1.x, pos1.y, w, h);
//     // @ts-ignore
//     ellipse(pos2.x, pos2.y, w, h);
//     // @ts-ignore
//     ellipse(pos3.x, pos3.y, w, h);
//     // @ts-ignore
//     ellipse(pos4.x, pos4.y, w, h);
//     // @ts-ignore
//     ellipse(pos5.x, pos5.y, w, h);
//     // @ts-ignore
//     pop();
// }

function loss(xr: Xor, t: number[][]): number {
    // if(ti.rows != to.rows || to.cols != xr.mA2.cols){
    //     console.error('loss error 1: rows or cols do not match');
    // } 
    // const n = ti.rows;
    // let l = 0;
    // for(let i = 0; i < n; ++i){
    //     const x: Matrix = matrixRow(ti,i);
    //     const y: Matrix = matrixRow(to,i)
    //
    //     matrixCopy(xr.mA0, x);
    //     xorForward(xr);
    //
    //     const m = to.cols;
    //     for(let j = 0; j < m; ++j){
    //         const dist = xr.mA2.data[0][j] - y.data[0][j];
    //         l += dist * dist;
    //     }
    // }
    // console.log(`l/n = ${l/n}`);
    // return l / n;

    let result: number = 0;
    for(let i = 0; i < t.length; ++i){
        const x1: number = t[i][0]; 
        const x2: number = t[i][1];
        console.log(x1);
        console.log(x2);
        xr.mA0.data[i][0] = x1;
        xr.mA0.data[i][1] = x2;
        xorForward(xr);
        const y = xr.mA2.data[0][0];
        const d = y - t[i][2];
        result += d * d;
    }
    return result;
}

function wiggle(xr: Xor, xrMat: Matrix, gradientMat: Matrix, epsilon: number, t: number[][]): void {
    let saved: number;
    const l = loss(xr, t);
    for(let i = 0; i < xrMat.rows; ++i){
        for(let j = 0; j < xrMat.cols; ++j){
            saved = xrMat.data[i][j];
            xrMat.data[i][j] += epsilon;
            console.log("here3");
            console.log((loss(xr, t)));
            console.log(l);
            gradientMat.data[i][j] = (loss(xr, t)-l)/epsilon;
            xrMat.data[i][j] = saved;
        }
    }
}

function finiteDiff(xr: Xor, gradient: Xor, epsilon: number, t: number[][]): void {
    wiggle(xr, xr.weightsLayer1, gradient.weightsLayer1, epsilon, t);
    console.log("here");
    console.table(xr.weightsLayer1.data);
    console.log("here2");
    console.table(gradient.weightsLayer1.data);
    // wiggle(xr, xr.biasesLayer1, gradient.biasesLayer1, epsilon, ti, to);
    // wiggle(xr, xr.weightsLayer2, gradient.weightsLayer2, epsilon, ti, to);
    // wiggle(xr, xr.biasesLayer2, gradient.biasesLayer2, epsilon, ti, to);
}

function wiggleSub(xrMat: Matrix, gradientMat: Matrix, rate: number): void {
    for(let i = 0; i < xrMat.rows; ++i){
        for(let j = 0; j < xrMat.cols; ++j){
            xrMat.data[i][j] -= gradientMat.data[i][j] * rate;
        }
    }
}

interface Xor {
    mA0: Matrix;
    weightsLayer1: Matrix; biasesLayer1: Matrix; mA1: Matrix;
    weightsLayer2: Matrix; biasesLayer2: Matrix; mA2: Matrix;
}

function xorLearn(xr: Xor, gr: Xor, rate: number): void {
    wiggleSub(xr.weightsLayer1, gr.weightsLayer1, rate);
    wiggleSub(xr.biasesLayer1, gr.biasesLayer1, rate);
    wiggleSub(xr.weightsLayer2, gr.weightsLayer2, rate);
    wiggleSub(xr.biasesLayer2, gr.biasesLayer2, rate);
}

function xorRandomize(xr: Xor): void {
    matrixRandomize(xr.weightsLayer1, 0, 1);
    matrixRandomize(xr.biasesLayer1, 0, 1);
    matrixRandomize(xr.weightsLayer2, 0, 1);
    matrixRandomize(xr.biasesLayer2, 0, 1);
}

function xorCreate(x: Matrix, w1: Matrix, b1: Matrix, a1: Matrix, w2: Matrix, b2: Matrix, a2: Matrix): Xor {
    return {mA0: x, weightsLayer1: w1, biasesLayer1: b1, mA1: a1, weightsLayer2: w2, biasesLayer2: b2, mA2: a2};
}

function xorForward(xr: Xor): void {
    matrixMult(xr.mA1, xr.mA0, xr.weightsLayer1);
    matrixSum(xr.mA1, xr.biasesLayer1);
    matrixSigmoidf(xr.mA1);

    matrixMult(xr.mA2, xr.mA1, xr.weightsLayer2); 
    matrixSum(xr.mA2, xr.biasesLayer2);
    matrixSigmoidf(xr.mA2);
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

function matrixSlice(input: number[][], startIndex: number, cols: number): number[][] {
    return input.map(row => row.slice(startIndex, startIndex + cols));
}

function matrixRow(mat: Matrix, row: number): Matrix {
    return {rows: 1, cols: mat.cols, data: [[...mat.data[row]]]} 
}

function matrixCopy(dst: Matrix, src: Matrix): void {
   if(dst.cols !== src.cols || dst.rows !== src.rows){
        console.error('copy error 1: for params, rows or cols do not match');       
   }
   dst = {rows: src.rows, cols: src.cols, data: [...src.data]}; 
}

function matrixMult(dst: Matrix, a: Matrix, b: Matrix): void {
    if(a.cols !== b.rows){
        console.error('mult error 1: for param2 and param3, the rows do not match'); 
    }

    if(dst.rows !== a.rows || dst.cols !== b.cols){
        console.error('mult error 2: for params, either the rows of param1 and param2 or cols of param1 and param3 do not match'); 
    }

    const n = a.cols;
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

// function matrixSub(dst: Matrix, a: Matrix): void {
// }

function matrixRandomize(mat: Matrix, low: number, high: number): void {
    const seed: number[] = cyrb128(Math.random().toString());
    const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);
    for(let i = 0; i < mat.rows; ++i){
        for(let j = 0; j < mat.cols; ++j){
            mat.data[i][j] = rnd() * (high - low) + low;
        }
    }
}

function matrixPrint(mat: Matrix, str: string): void {
    console.log(str + ':');
    console.table(mat.data);
}

function matrixSigmoidf(mat: Matrix): void {
    for(let i = 0; i < mat.data.length; ++i){
        for(let j = 0; j < mat.data[i].length; ++j){
            mat.data[i][j] = sigmoidf(mat.data[i][j]);
        }
    }
}

function sigmoidf(x: number): number {

    return 1 / (1 + Math.exp(-x));
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
