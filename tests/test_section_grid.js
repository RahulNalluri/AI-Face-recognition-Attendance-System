// Lightweight DOM contract tests for the actual grid script; no browser or packages needed.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.events = {}; this.value = ''; this.textContent = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  add(node) { this.append(node); }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, listener) { this.events[name] = listener; }
}

test('lunch is one non-editable column and moving it preserves subjects', () => {
  const nodes = {};
  const document = {
    getElementById(id) { return nodes[id] ??= new Element('div'); },
    createElement(tag) { return new Element(tag); },
  };
  const grid = {day_start:'09:30', lunch_after:3,
    periods:Array.from({length:6}, (_,i)=>({duration:60,break_after:i===2?60:0,repeat:0,window:10})),
    subjects:Array.from({length:6}, ()=>['AI','','','','',''])};
  document.getElementById('section-grid-data').textContent = JSON.stringify(grid);
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/static/section_grid.js'), 'utf8'), {
    document, Option: class extends Element {constructor(text, value) {super('option');this.textContent=text;this.value=value;}},
    window: {confirm:()=>true},
  });
  const header = () => nodes['grid-head'].children[0].children;
  assert.equal(header().length, 8); // day + six periods + lunch
  assert.equal(header()[4].textContent, 'Lunch break');
  assert.equal(header()[4].children[0].textContent, '12:30–13:30');
  assert.equal(header()[5].children[0].textContent, '13:30–14:30');
  assert.equal(nodes['period-settings'].children[2].children.length, 5);
  assert.equal(nodes['period-settings'].children[2].children[2].children[0].disabled, true);
  assert.ok(nodes['grid-body'].children.every(row=>row.children[4].children.length===1));
  nodes['lunch-after'].value = '2';
  nodes['lunch-after'].events.change();
  assert.equal(header()[3].textContent, 'Lunch break');
  assert.equal(header()[3].children[0].textContent, '11:30–12:30');
  assert.equal(nodes['grid-body'].children[0].children[1].children[0].value, 'AI');
  assert.equal(nodes['period-settings'].children[2].children[2].children[0].value, 0);
  assert.equal(nodes['period-settings'].children[1].children[2].children[0].value, 60);
});
