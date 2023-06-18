const orGate: number[][] = [
    [0,0,0],
    [1,0,1],
    [0,1,1],
    [1,1,1],
];

const andGate: number[][] = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [1,1,1],
];

const nandGate: number[][] = [
    [0,0,1],
    [1,0,1],
    [0,1,1],
    [1,1,0],
];

const xorGate: number[][] = [
    [0,0,0],
    [1,0,1],
    [0,1,1],
    [1,1,0],
];

const norGate: number[][] = [
    [0,0,1],
    [1,0,0],
    [0,1,0],
    [1,1,0],
];

const train: number[][] = [...nandGate];
const trainCount: number = train.length*train[0].length;

function main(): void{

    const epsilon = 1e-1;
    const rate = 1e-1;

    const model: Gate = new Gate();
    model.log();

    console.log('-----------------');
    console.log(`loss before = ${model.loss()}`);

    for(let i = 0; i < 100*1000; ++i){
        const gradientXor: Gate = model.finiteDiff(epsilon);
        model.learn(gradientXor, rate);
    }
    console.log('-----------------');

    console.log(`loss after = ${model.loss()}`);

    console.log('-----------------');
    console.log('Neuron 1:');
    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            console.log(`${i} ? ${j} = ${model.sigmoidf(model.neurons[0][0] * i + model.neurons[0][1] * j + model.neurons[0][2])}`);
        }
    }

    console.log('-----------------');
    console.log('Neuron 2:');
    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            console.log(`${i} ? ${j} = ${model.sigmoidf(model.neurons[1][0] * i + model.neurons[1][1] * j + model.neurons[1][2])}`);
        }
    }

    console.log('-----------------');
    console.log('Neuron 3:');
    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            console.log(`${i} ? ${j} = ${model.sigmoidf(model.neurons[2][0] * i + model.neurons[2][1] * j + model.neurons[2][2])}`); }
    }

    console.log('-----------------');
    console.log('Xor Gate:');
    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            console.log(`${i} ? ${j} = ${model.forward(i, j)}`);
        }
    }
    console.log('-----------------');

}

class Gate {

    neurons: number[][];

    // orW1: number;
    // orW2: number;
    // orBias: number;
    // andW1: number;
    // andW2: number;
    // andBias: number;
    // nandW1: number;
    // nandW2: number;
    // nandBias: number;

    constructor(){
        const seed: number[] = cyrb128(Math.random().toString());
        const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);

        const numNeurons = 3;
        const numConnections = 3;
        this.neurons = [];
        for(let i = 0; i < numNeurons; ++i){
            this.neurons[i] = [];
            for(let j = 0; j < numConnections; ++j){
                const randomNumber = rnd();
                this.neurons[i].push(randomNumber);
            }
        }
    }

    loss(): number {
        let result: number = 0;
        for (let i = 0; i < train.length; ++i) {
            const x1: number = train[i][0];
            const x2: number = train[i][1];
            const y: number = this.forward(x1, x2);
            const d: number = y - train[i][2];
            result += d*d;
        }
        result /= trainCount;
        return result;
    }

    learn(gradient: Gate, rate: number): void{
        for(let i = 0; i < this.neurons.length; ++i){
            for(let j = 0; j < this.neurons[i].length; ++j){
                this.neurons[i][j] -= gradient.neurons[i][j] * rate;
            }
        }
    }

    forward(x: number, y: number): number {
        const a: number = this.sigmoidf(this.neurons[0][0] * x + this.neurons[0][1] * y + this.neurons[0][2]);
        const b: number = this.sigmoidf(this.neurons[1][0] * x + this.neurons[1][1] * y + this.neurons[1][2]);
        return this.sigmoidf(a*this.neurons[2][0] + b*this.neurons[2][1] + this.neurons[2][2]);
    }

    sigmoidf(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    finiteDiff(epsilon: number): Gate{
        let gradientXor: Gate = new Gate();
        const l = this.loss();

        for(let i = 0; i < this.neurons.length; ++i){
            for(let j = 0; j < this.neurons[i].length; ++j){
                const saved: number = this.neurons[i][j];
                this.neurons[i][j] += epsilon;
                gradientXor.neurons[i][j] = (this.loss() - l)/epsilon;
                this.neurons[i][j] = saved;
            }
        }
        return gradientXor;
    }

    copy(model: Gate): void{
       for(let i = 0; i < this.neurons.length; ++i){
            for(let j = 0; j < this.neurons[i].length; ++j){
                this.neurons[i][j] = model.neurons[i][j];
            }
        }
    }

    log(): void{
        for(let i = 0; i < this.neurons.length; ++i){
            console.log(`W1 = ${this.neurons[i][0]}, W2 = ${this.neurons[i][1]}, Bias = ${this.neurons[i][2]}`);
        }
    }
};


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
