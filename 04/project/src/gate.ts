


type Xor = {
    x: Matrix ; 
              
    w1: Matrix;
    b1: Matrix;
    a1: Matrix;
              
    w2: Matrix;
    b2: Matrix;
    a2: Matrix;

    expected?: number[];
}

function xorCreate(x: Matrix, w1: Matrix, b1: Matrix, a1: Matrix, w2: Matrix, b2: Matrix, a2: Matrix): Xor {
    return {x: x, w1: w1, b1: b1, a1: a1, w2: w2, b2: b2, a2: a2};
}

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
