
let xor: Xor;
let gradient: Xor;

function main(){
    xor = xorCreate(
        matrixCreate(1, 2),

        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),

        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1),
    )

    gradient = xorCreate(
        matrixCreate(1, 2),

        matrixCreate(2, 2),
        matrixCreate(1, 2),
        matrixCreate(1, 2),

        matrixCreate(2, 1),
        matrixCreate(1, 1),
        matrixCreate(1, 1),
    )

    // or
    xor.expected = [
        0,0,0,
        1,0,1,
        0,1,1,
        1,1,1,
    ];

    // and
    xor.expected = [
        0,0,0,
        1,0,0,
        0,1,0,
        1,1,1,
    ];

    // nand
    xor.expected = [
        0,0,1,
        1,0,1,
        0,1,1,
        1,1,0,
    ];

    // nor
    xor.expected = [
        0,0,1,
        1,0,0,
        0,1,0,
        1,1,0,
    ];

    // xor
    xor.expected = [
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    ];

    const stride = 3;

    const tiArr: number[] = matrixSlice(xor.expected,xor.expected.length/stride,2,stride,0);
    const toArr: number[] = matrixSlice(xor.expected,xor.expected.length/stride,1,stride,2);

    const ti: Matrix = { rows: xor.expected.length/stride, cols: 2, 
        stride: stride, samples: [...tiArr] };

    const to: Matrix = { rows: xor.expected.length/stride, cols: 1, 
        stride: stride, samples: [...toArr] };

    // matrixPrint(ti, 'ti');
    // matrixPrint(to, 'to');

    matrixRandomize(xor.w1, 0, 1);
    matrixRandomize(xor.b1, 0, 1);
    matrixRandomize(xor.w2, 0, 1);
    matrixRandomize(xor.b2, 0, 1);

    const epsilon = 1e-1;
    const rate = 1e-1;
    // console.log(`loss = ${xorLoss(xor, ti, to)}`);
    for(let i = 0; i < 100*1000; ++i){
        xorFiniteDiff(xor, gradient, epsilon, ti, to);
        xorLearn(xor, gradient, rate);
    }
    // console.log(`loss = ${xorLoss(xor, ti, to)}`);

    for(let i = 0; i < 2; ++i){
        for(let j = 0; j < 2; ++j){
            xor.x.samples = [i, j];
            xor = xorForward(xor);
            const y = xor.a2.samples[0];
            console.log(`${i} ^ ${j} = ${y}`);
        }
    }
}

main();
