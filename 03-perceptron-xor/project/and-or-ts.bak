
// OR GATE TRAIN SET
// const train: number[][] = [
//     [0,0,0],
//     [1,0,1],
//     [0,1,1],
//     [1,1,1],
// ];

// AND GATE TRAIN SET
const train: number[][] = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [1,1,1],
];

const trainCount: number = train.length*train[0].length;

function main(): void{
    // const seed: number[] = cyrb128(Date.now().toString());
    const seed: number[] = cyrb128('69');
    const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);
    let w1: number = rnd();
    let w2: number = rnd();
    let bias: number = rnd();

    const epsilon: number = 1e-1;
    const rate: number = 1e-1;

    for(let i = 0; i < 1000000; ++i){
        const l: number = loss(w1, w2, bias);
        const dw1: number = (loss(w1 + epsilon, w2, bias) - l)/epsilon;
        const dw2: number = (loss(w1, w2 + epsilon, bias) - l)/epsilon;
        const db: number = (loss(w1, w2, bias + epsilon) - l)/epsilon;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        bias -= rate*db;
    }
    console.log(`w1 = ${w1}, w2 = ${w2}, bias = ${bias}, loss: ${loss(w1, w2, trainCount)}`);

    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
           console.log(`${i} | ${j} = ${sigmoidf(i*w1 + j*w2 + bias)}`); 
        }
    }
}

function loss(w1: number, w2: number, bias:number): number {
    let result: number = 0;
    for (let i = 0; i < train.length; ++i) {
        const x1: number = train[i][0];
        const x2: number = train[i][1];
        const y: number = sigmoidf(x1*w1 + x2*w2 + bias);
        const d: number = y - train[i][2];
        result += d*d;
    }
    result /= trainCount;
    return result;
}

function sigmoidf(x: number): number {
    return 1 / (1 + Math.exp(-x));
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

main();
