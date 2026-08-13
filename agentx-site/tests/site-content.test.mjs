import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { test } from "node:test";

const appRoot = new URL("../app/", import.meta.url);
const publicRoot = new URL("../public/", import.meta.url);

test("analysis narrative includes headline metrics and cache caveat", async () => {
  const page = await readFile(new URL("page.tsx", appRoot), "utf8");
  assert.match(page, /98\.3%/);
  assert.match(page, /42\.5%/);
  assert.match(page, /theoretical reuse available/);
  assert.match(page, /Acceptance rates require separate/);
});

test("required visual assets are present", async () => {
  await Promise.all([
    access(new URL("agentx-social.png", publicRoot)),
    access(new URL("plots\/joint_isl_osl.png", publicRoot)),
    access(new URL("plots\/representative_complex_timeline.png", publicRoot)),
  ]);
});
