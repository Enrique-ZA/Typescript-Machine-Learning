
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
