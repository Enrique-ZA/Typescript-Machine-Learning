
type Gate = {
    x: Matrix ; 
              
    w1: Matrix;
    b1: Matrix;
    a1: Matrix;
              
    w2: Matrix;
    b2: Matrix;
    a2: Matrix;

    expected?: number[];
}

function gateCreate(x: Matrix, w1: Matrix, b1: Matrix, a1: Matrix, w2: Matrix, b2: Matrix, a2: Matrix): Gate {
    return {x: x, w1: w1, b1: b1, a1: a1, w2: w2, b2: b2, a2: a2};
}

function gateLoss(gate: Gate, ti: Matrix, to: Matrix): number {
    if(ti.rows != to.rows || to.cols != gate.a2.cols){
        throw new Error("loss error: ti.rows != to.rows || to.cols != gate.a2.cols");
    }
    let result = 0;
    for(let i = 0; i < ti.rows; ++i){
        const rowMatrix1 = matrixRow(ti, i);
        const rowMatrix2 = matrixRow(to, i);

        gate.x = matrixCopy(gate.x, rowMatrix1);
        gate = gateForward(gate);

        for(let j = 0; j < to.cols; ++j){
            const dist = gate.a2.samples[gate.a2.cols * 0 + j] - rowMatrix2.samples[rowMatrix2.cols * 0 + j];
            result += (dist * dist);
        }
    }
    return (result /= ti.rows);
}

function gateLearn(gate: Gate, gradient: Gate, rate: number){
    for(let i = 0; i < gate.w1.rows; ++i){
        for(let j = 0; j < gate.w1.cols; ++j){
            gate.w1.samples[gate.w1.cols * i + j] -= gradient.w1.samples[gradient.w1.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < gate.b1.rows; ++i){
        for(let j = 0; j < gate.b1.cols; ++j){
            gate.b1.samples[gate.b1.cols * i + j] -= gradient.b1.samples[gradient.b1.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < gate.w2.rows; ++i){
        for(let j = 0; j < gate.w2.cols; ++j){
            gate.w2.samples[gate.w2.cols * i + j] -= gradient.w2.samples[gradient.w2.cols * i + j] * rate;
        }
    }

    for(let i = 0; i < gate.b2.rows; ++i){
        for(let j = 0; j < gate.b2.cols; ++j){
            gate.b2.samples[gate.b2.cols * i + j] -= gradient.b2.samples[gradient.b2.cols * i + j] * rate;
        }
    }
}

function gateFiniteDiff(gate: Gate, gradient: Gate, epsilon: number, ti: Matrix, to: Matrix): void {
    let saved;
    const loss = gateLoss(gate, ti, to);

    for(let i = 0; i < gate.w1.rows; ++i){
        for(let j = 0; j < gate.w1.cols; ++j){
            saved = gate.w1.samples[gate.w1.cols * i + j];
            gate.w1.samples[gate.w1.cols * i + j] += epsilon;
            gradient.w1.samples[gradient.w1.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon;
            gate.w1.samples[gate.w1.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < gate.b1.rows; ++i){
        for(let j = 0; j < gate.b1.cols; ++j){
            saved = gate.b1.samples[gate.b1.cols * i + j];
            gate.b1.samples[gate.b1.cols * i + j] += epsilon;
            gradient.b1.samples[gradient.b1.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon;
            gate.b1.samples[gate.b1.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < gate.w2.rows; ++i){
        for(let j = 0; j < gate.w2.cols; ++j){
            saved = gate.w2.samples[gate.w2.cols * i + j];
            gate.w2.samples[gate.w2.cols * i + j] += epsilon;
            gradient.w2.samples[gradient.w2.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon;
            gate.w2.samples[gate.w2.cols * i + j] = saved;
        }
    }

    for(let i = 0; i < gate.b2.rows; ++i){
        for(let j = 0; j < gate.b2.cols; ++j){
            saved = gate.b2.samples[gate.b2.cols * i + j];
            gate.b2.samples[gate.b2.cols * i + j] += epsilon;
            gradient.b2.samples[gradient.b2.cols * i + j] = (gateLoss(gate, ti, to)-loss)/epsilon;
            gate.b2.samples[gate.b2.cols * i + j] = saved;
        }
    }
}

function gateForward(gate: Gate): Gate {
    gate.a1 = matrixMult     (gate.a1, gate.x,     gate.w1);
    gate.a1 = matrixSum      (gate.a1, gate.b1);
    gate.a1 = matrixSigmoidf (gate.a1);

    gate.a2 = matrixMult     (gate.a2, gate.a1,    gate.w2);
    gate.a2 = matrixSum      (gate.a2, gate.b2);
    gate.a2 = matrixSigmoidf (gate.a2);
    return gate;
}
