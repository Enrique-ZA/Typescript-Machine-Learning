
const train: number[][] = [
    [0,0],
    [1,2],
    [2,4],
    [3,6],
    [4,8],
    [5,10],
];

// y = x*w;
// const seed: number[] = cyrb128(Date.now().toString());

function main(): void{
    const seed: number[] = cyrb128('69');
    const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);
    let w: number = rnd() * 10.0;
    let bias: number = rnd() * 5.0;
    const trainCount: number = train.length*train[0].length;

    const epsilon: number = 1e-3;
    const rate: number = 1e-3;

    console.log(loss(w, bias, trainCount));
    for(let i = 0; i < 20000; ++i){
        // const dloss: number = (loss(w + epsilon, 0, trainCount) - loss(w, 0, trainCount)) / epsilon;
        const dw: number = (loss(w + epsilon, bias, trainCount) - loss(w, bias, trainCount)) / epsilon;
        const db: number = (loss(w, bias + epsilon, trainCount) - loss(w, bias, trainCount)) / epsilon;
        w -= rate*dw;
        bias -= rate*db;
        console.log(`${loss(w, bias, trainCount)}, w: ${w}, b: ${bias}`);
    }
    console.log('---------------------------------');
    console.log(w);
    console.log(bias);
}

function loss(w: number, bias: number, trainCount: number): number {
    let result: number = 0;
    for (let i = 0; i < train.length; ++i) {
        const x = train[i][0];
        const y = x*w + bias;
        const d = y - train[i][1];
        result += d*d;
    }
    result /= trainCount;
    return result;
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

// cost/loss function end
main();
