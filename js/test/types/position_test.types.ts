// Type-level checks of positionTest's overloads, compiled by
// test/validation.test.ts with the TypeScript compiler (never run).
import type { NdArray } from "../../src/ndarray.ts";
import {
  positionTest, BacktestPriceTarget, BacktestPortfolioTarget,
  type PositionTestResult, type PositionFamilyResult,
} from "../../src/index.ts";

type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
const is = <T>(_: T) => {};

declare const series: Float64Array, nums: number[], nd: NdArray, rows: number[][];
const eng = BacktestPriceTarget();

const one = positionTest(eng, series, [series]);
is<Equals<typeof one, PositionTestResult>>(true);
const oneNd = positionTest(eng, { data: series, shape: [10] }, [nums]);
is<Equals<typeof oneNd, PositionTestResult>>(true);
const fam = positionTest(eng, [series, nums], [series]);
is<Equals<typeof fam, PositionFamilyResult>>(true);
const famNd = positionTest(eng, { data: series, shape: [5, 2] }, [series]);
is<Equals<typeof famNd, PositionFamilyResult>>(true);
const either = positionTest(eng, nd, [series]);
is<Equals<typeof either, PositionTestResult | PositionFamilyResult>>(true);
const book = positionTest(BacktestPortfolioTarget(), rows, [rows]);
is<Equals<typeof book, PositionTestResult>>(true);
const bookNd = positionTest(BacktestPortfolioTarget(), nd, [nd]);
is<Equals<typeof bookNd, PositionTestResult>>(true);

// @ts-expect-error a K-strategy result is not one strategy's
is<Equals<typeof famNd, PositionTestResult>>(true);
// @ts-expect-error the markets must be arrays
positionTest(eng, series, series);
