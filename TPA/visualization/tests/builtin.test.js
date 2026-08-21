'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const builtin = require('../registry/schemas/builtin.js');


test('builtin.js 镜像与 schema/*.json 规范源保持一致', () => {
  const dir = path.join(__dirname, '..', 'schema');
  const fromDisk = fs.readdirSync(dir)
    .filter((f) => f.endsWith('.schema.json'))
    .map((f) => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf-8')));
  assert.strictEqual(builtin.length, fromDisk.length);
  for (const s of builtin) {
    const disk = fromDisk.find((d) => d.id === s.id);
    assert.ok(disk, `schema ${s.id} 缺少磁盘镜像`);
    assert.deepStrictEqual(s, disk);
  }
});
