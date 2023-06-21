function xorLoss(xor: Xor, ti: Matrix, to: Matrix): number {
    if(ti.rows != to.rows || to.cols != xor.a2.cols){
        throw new Error("loss error: ti.rows != to.rows || to.cols != xor.a2.cols");
    }
    let result = 0;
    for(let i = 0; i < ti.rows; ++i){
        const rowMatrix1 = matrixRow(ti, i);
        const rowMatrix2 = matrixRow(to, i);

        xor.x = matrixCopy(xor.x, rowMatrix1);
        xor = xorForward(xor);

        for(let j = 0; j < to.cols; ++j){
            const dist = xor.a2.samples[xor.a2.cols * 0 + j] - rowMatrix2.samples[rowMatrix2.cols * 0 + j];
            result += (dist * dist);
        }
    }
    return (result /= ti.rows);
}

function xorLearn(xor: Xor, gradient: Xor, rate: number){
    for(let i = 0; i < xor.w1.rows; ++i){
        for(let j = 0; j < xor.w1.cols; ++j){
            xor.w1.samples[xor.w1.cols * i + j] -= gradient.w1.samples[gradient.w1.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < xor.b1.rows; ++i){
        for(let j = 0; j < xor.b1.cols; ++j){
            xor.b1.samples[xor.b1.cols * i + j] -= gradient.b1.samples[gradient.b1.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < xor.w2.rows; ++i){
        for(let j = 0; j < xor.w2.cols; ++j){
            xor.w2.samples[xor.w2.cols * i + j] -= gradient.w2.samples[gradient.w2.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < xor.b2.rows; ++i){
        for(let j = 0; j < xor.b2.cols; ++j){
            xor.b2.samples[xor.b2.cols * i + j] -= gradient.b2.samples[gradient.b2.cols * i + j] * rate;
        }
    }
}

function xorFiniteDiff(xor: Xor, gradient: Xor, epsilon: number, ti: Matrix, to: Matrix): void {
    let saved;
    const loss = xorLoss(xor, ti, to);

    for(let i = 0; i < xor.w1.rows; ++i){
        for(let j = 0; j < xor.w1.cols; ++j){
            saved = xor.w1.samples[xor.w1.cols * i + j];
            xor.w1.samples[xor.w1.cols * i + j] += epsilon;
            gradient.w1.samples[gradient.w1.cols * i + j] = (xorLoss(xor, ti, to)-loss)/epsilon;
            xor.w1.samples[xor.w1.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < xor.b1.rows; ++i){
        for(let j = 0; j < xor.b1.cols; ++j){
            saved = xor.b1.samples[xor.b1.cols * i + j];
            xor.b1.samples[xor.b1.cols * i + j] += epsilon;
            gradient.b1.samples[gradient.b1.cols * i + j] = (xorLoss(xor, ti, to)-loss)/epsilon;
            xor.b1.samples[xor.b1.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < xor.w2.rows; ++i){
        for(let j = 0; j < xor.w2.cols; ++j){
            saved = xor.w2.samples[xor.w2.cols * i + j];
            xor.w2.samples[xor.w2.cols * i + j] += epsilon;
            gradient.w2.samples[gradient.w2.cols * i + j] = (xorLoss(xor, ti, to)-loss)/epsilon;
            xor.w2.samples[xor.w2.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < xor.b2.rows; ++i){
        for(let j = 0; j < xor.b2.cols; ++j){
            saved = xor.b2.samples[xor.b2.cols * i + j];
            xor.b2.samples[xor.b2.cols * i + j] += epsilon;
            gradient.b2.samples[gradient.b2.cols * i + j] = (xorLoss(xor, ti, to)-loss)/epsilon;
            xor.b2.samples[xor.b2.cols * i + j] = saved;
        }
    }
}

function xorForward(xor: Xor): Xor {
    xor.a1 = matrixMult     (xor.a1, xor.x,     xor.w1);
    xor.a1 = matrixSum      (xor.a1, xor.b1);
    xor.a1 = matrixSigmoidf (xor.a1);

    xor.a2 = matrixMult     (xor.a2, xor.a1,    xor.w2);
    xor.a2 = matrixSum      (xor.a2, xor.b2);
    xor.a2 = matrixSigmoidf (xor.a2);
    return xor;
}

function matrixSlice(arr: number[], rows: number, cols: number, step: number, start: number): number[] {
    if (arr === undefined) throw new Error("array is undefined");
    let test: number[] = [];
    let index = start;
    for(let i = 0; i < rows; i++){
        for(let j = 0; j < cols; ++j){
            if (index < arr.length) {
                test.push(arr[index]);
                index++;
            }
        }
        index += step - cols; // Skip 'step - cols' indices after each row
    }
    return test;
}

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
    console.log(`loss = ${xorLoss(xor, ti, to)}`);
    for(let i = 0; i < 100*1000; ++i){
        xorFiniteDiff(xor, gradient, epsilon, ti, to);
        xorLearn(xor, gradient, rate);
    }
    console.log(`loss = ${xorLoss(xor, ti, to)}`);

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
