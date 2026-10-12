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

// NdArray series markets
declare const ndSeries: NdArray & { shape: [number] }, ndPanel: NdArray & { shape: [number, number] };
const famNdMarket = positionTest(eng, { data: series, shape: [5, 2] }, [{ data: series, shape: [5] }]);
is<Equals<typeof famNdMarket, PositionFamilyResult>>(true);
const colsNdMarket = positionTest(eng, [series, nums], [ndSeries]);
is<Equals<typeof colsNdMarket, PositionFamilyResult>>(true);
const oneNdMarket = positionTest(eng, series, [ndSeries]);
is<Equals<typeof oneNdMarket, PositionTestResult>>(true);
const oneAnyNd = positionTest(eng, series, [nd]);        // a (T,) target is one strategy
is<Equals<typeof oneAnyNd, PositionTestResult>>(true);
const oneNdBoth = positionTest(eng, ndSeries, [nd, nums]);
is<Equals<typeof oneNdBoth, PositionTestResult>>(true);
const famMixed = positionTest(eng, ndPanel, [ndSeries, series]);
is<Equals<typeof famMixed, PositionFamilyResult>>(true);
// NdArray panel markets: a portfolio engine
const bookPanel = positionTest(BacktestPortfolioTarget(), ndPanel, [ndPanel, rows]);
is<Equals<typeof bookPanel, PositionTestResult>>(true);
const bookRows = positionTest(BacktestPortfolioTarget(), rows, [ndPanel]);
is<Equals<typeof bookRows, PositionTestResult>>(true);
// the rank is not in the types: the union
const unknownMarket = positionTest(eng, ndPanel, [nd]);
is<Equals<typeof unknownMarket, PositionTestResult | PositionFamilyResult>>(true);
const unknownBoth = positionTest(eng, nd, [nd]);
is<Equals<typeof unknownBoth, PositionTestResult | PositionFamilyResult>>(true);

// @ts-expect-error a K-strategy result is not one strategy's
is<Equals<typeof famNd, PositionTestResult>>(true);
// @ts-expect-error the markets must be arrays
positionTest(eng, series, series);
