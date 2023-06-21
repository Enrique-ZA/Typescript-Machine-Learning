
type Matrix = {
    rows: number;
    cols: number;
    stride?: number;
    samples: number[];
}

function matrixCreate(numRows: number, numCols: number){
    let arr: number[] = [];
    for (let i = 0; i < numRows; ++i) {
        for (let j = 0; j < numCols; ++j) {
            arr[numCols * i + j] = 0;
        }
    }
    return {rows: numRows, cols: numCols, stride: numCols, samples: arr};
}

function matrixPrint(mat: Matrix, name: string): void {
	let arr: number[][] = [];
    for (let i = 0; i < mat.rows; ++i) {
        arr[i] = [];
        for (let j = 0; j < mat.cols; ++j) {
            arr[i][j] = mat.samples[mat.cols * i + j];
        }
    }
    console.log(name + ':');
	console.table(arr);
}

function matrixRandomize(mat: Matrix, low: number, high: number): Matrix {
    const seed: number[] = cyrb128(Math.random().toString());
    const rnd: () => number = sfc32(seed[0], seed[1], seed[2], seed[3]);
    for(let i = 0; i < mat.rows; ++i){
        for(let j = 0; j < mat.cols; ++j){
            mat.samples[mat.cols * i + j] = rnd() * (high - low) + low;
        }
    }
    return mat;
}

function matrixSum(org: Matrix, other: Matrix): Matrix {
    if (other.rows !== org.rows || other.cols !== org.cols) {
        throw new Error('matrix sum error: other and org must have the same size');
    }
    for(let i = 0; i < other.rows; ++i){
        for(let j = 0; j < other.cols; ++j){
            org.samples[other.cols * i + j] += other.samples[other.cols * i + j];
        }
    }
    return org;
}

function matrixFill(mat: Matrix, num: number): Matrix {
    for(let i = 0; i < mat.rows; ++i){
        for(let j = 0; j < mat.cols; ++j){
            mat.samples[mat.cols * i + j] = num;
        }
    }
    return mat;
}

function matrixMult(dst: Matrix, a: Matrix, b: Matrix): Matrix {
    if(a.cols !== b.rows){
        throw new Error('mult error 1: for param2 and param3, the rows do not match'); 
    }

    if(dst.rows !== a.rows || dst.cols !== b.cols){
        throw new Error('mult error 2: for params, either the rows of param1 and param2 or cols of param1 and param3 do not match'); 
    }

    const n = a.cols;
    for(let i = 0; i < dst.rows; ++i){
        for(let j = 0; j < dst.cols; ++j){
            dst.samples[dst.cols * i + j] = 0;
            for(let k = 0; k < n; ++k){
                dst.samples[dst.cols * i + j] += a.samples[a.cols * i + k] * b.samples[b.cols * k + j];
            }
        }
    }
    return dst;
}

function matrixSigmoidf(mat: Matrix): Matrix {
    for (let i = 0; i < mat.rows; ++i) {
        for (let j = 0; j < mat.cols; ++j) {
            mat.samples[mat.cols * i + j] = sigmoidf(mat.samples[mat.cols * i + j]);
        }
    }
    return mat;
}

function matrixRow(mat: Matrix, row: number): Matrix {
    let startIndex = row * mat.cols;
    let endIndex = startIndex + mat.cols;
    let rowSamples = mat.samples.slice(startIndex, endIndex);
    return {rows: 1, cols: mat.cols, stride: mat.cols, samples: rowSamples}; 
}

function matrixCopy(dst: Matrix, src: Matrix): Matrix {
    if(dst.rows !== src.rows || dst.cols !== src.cols){
        throw new Error("copy error: matrices don't match"); 
    }
    dst.samples = [...src.samples];
    return dst;
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
