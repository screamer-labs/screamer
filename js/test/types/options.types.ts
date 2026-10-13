// Type-level checks of the generated options-object overloads, compiled by
// test/ops.test.ts with the TypeScript compiler (never run).
import {
  EwMean, RollingMean, MovingAverage, SchmittTrigger, Abs,
  type EwMeanOptions, type ScreamerOp,
} from "../../src/index.ts";

const is = <T>(_: T) => {};

// both call forms
is<ScreamerOp>(EwMean({ span: 20 }));
is<ScreamerOp>(EwMean(undefined, 20));
is<ScreamerOp>(RollingMean({ windowSize: 5, startPolicy: "expanding" }));
is<ScreamerOp>(RollingMean(5, "expanding"));
is<ScreamerOp>(MovingAverage({ taps: [0.5, 0.5] }));
is<ScreamerOp>(MovingAverage([0.5, 0.5]));
is<ScreamerOp>(SchmittTrigger({ lower: 1, upper: 3 }));
is<ScreamerOp>(Abs());
is<ScreamerOp>(Abs({}));
const opts: EwMeanOptions = { halflife: 5 };
is<ScreamerOp>(EwMean(opts));

// @ts-expect-error an unknown key
EwMean({ spn: 20 });
// @ts-expect-error the Python name is not the JavaScript one
RollingMean({ window_size: 5 });
// @ts-expect-error a number setting takes a number
EwMean({ span: "20" });
// @ts-expect-error a required setting is required
SchmittTrigger({ upper: 3 });
// @ts-expect-error an options object is the only argument
EwMean({ span: 20 }, 3);
// @ts-expect-error an op without settings has no options
Abs({ x: 1 });
// @ts-expect-error a number setting takes no array
RollingMean({ windowSize: [3] });

// null is "not given" exactly where Python takes None (an optional setting)
is<ScreamerOp>(EwMean(null, 20));
is<ScreamerOp>(EwMean({ com: null, span: 20 }));
// @ts-expect-error a setting with a fixed default refuses null
RollingMean(null);
// @ts-expect-error a setting with a fixed default refuses null
RollingMean({ windowSize: null });
// @ts-expect-error an op without settings takes no positional value
Abs(5);
