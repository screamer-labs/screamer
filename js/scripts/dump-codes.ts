// Prints the resample code tables of src/codes.ts as JSON, for the parity
// gate tests/test_dag_codes_sync.py:
//
//   node --import tsx scripts/dump-codes.ts
import { CODES } from "../src/codes.js";

process.stdout.write(JSON.stringify(CODES));
