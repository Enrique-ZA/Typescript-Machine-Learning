
let gate: Gate;
let gradient: Gate;

function main(){
    gate = gateCreate(
        matrixCreate(1, 2),

        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),

        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1),
    )

    gradient = gateCreate(
        matrixCreate(1, 2),

        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),

        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1),
    )

    // or
    gate.expected = [
        0,0,0,
        1,0,1,
        0,1,1,
        1,1,1,
    ];

    // and
    gate.expected = [
        0,0,0,
        1,0,0,
        0,1,0,
        1,1,1,
    ];

    // nand
    gate.expected = [
        0,0,1,
        1,0,1,
        0,1,1,
        1,1,0,
    ];

    // nor
    gate.expected = [
        0,0,1,
        1,0,0,
        0,1,0,
        1,1,0,
    ];

    // xnor
    gate.expected = [
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    ];

    // xor
    gate.expected = [
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    ];

    const stride = 3;

    const tiArr: number[] = matrixSlice(gate.expected,gate.expected.length/stride,2,stride,0);
    const toArr: number[] = matrixSlice(gate.expected,gate.expected.length/stride,1,stride,2);

    const ti: Matrix = { rows: gate.expected.length/stride, cols: 2, 
        stride: stride, samples: [...tiArr] };

    const to: Matrix = { rows: gate.expected.length/stride, cols: 1, 
        stride: stride, samples: [...toArr] };

    // matrixPrint(ti, 'ti');
    // matrixPrint(to, 'to');

    matrixRandomize(gate.w1, 0, 1);
    matrixRandomize(gate.b1, 0, 1);
    matrixRandomize(gate.w2, 0, 1);
    matrixRandomize(gate.b2, 0, 1);

    const epsilon = 1e-1;
    const rate = 1e-1;
    // console.log(`loss = ${gateLoss(gate, ti, to)}`);
    for(let i = 0; i < 50*1000; ++i){
        gateFiniteDiff(gate, gradient, epsilon, ti, to);
        gateLearn(gate, gradient, rate);
    }
    // console.log(`loss = ${gateLoss(gate, ti, to)}`);

    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            gate.x.samples = [i, j];
            gate = gateForward(gate);
            const y = gate.a2.samples[0];
            console.log(`${i} ^ ${j} = ${y}`);
        }
    }
}

main();
