// using p5js - https://p5js.org/download/

const colorWhite: [number, number, number, number] = [255, 255, 255, 255];
const colorBlack: [number, number, number, number] = [0, 0, 0, 255];

function preload(): void{
}

function setup(): void{
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
        for(let j = 0; j < numCols; ++j){
            mat[i][j] = -1;
        }
    }
    return {rows: numRows, cols: numCols, data: mat};
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
