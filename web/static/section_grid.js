// The form is posted as a single transaction with the section and its roster.
const sectionGrid = JSON.parse(document.getElementById('section-grid-data').textContent);
const gridDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const startField = document.getElementById('grid-start');
const countField = document.getElementById('period-count');
const lunchField = document.getElementById('lunch-after');
sectionGrid.lunch_after ??= null;
function gridCell(tag, text) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  return element;
}
function drawGrid() {
  const settings = document.getElementById('period-settings');
  const heading = document.getElementById('grid-head');
  const body = document.getElementById('grid-body');
  settings.replaceChildren(); heading.replaceChildren(); body.replaceChildren();
  lunchField.replaceChildren(new Option('No designated lunch break', ''));
  sectionGrid.periods.forEach((_, index) => lunchField.add(new Option(`After Period ${index + 1} — 60 minutes`, String(index + 1))));
  lunchField.value = sectionGrid.lunch_after === null ? '' : String(sectionGrid.lunch_after);
  sectionGrid.periods.forEach((period, index) => {
    const isLunch = sectionGrid.lunch_after === index + 1;
    if (isLunch) period.break_after = 60;
    const row = gridCell('tr'); row.append(gridCell('th', `Period ${index + 1}`));
    for (const [key, label, min] of [['duration', 'Duration', 1], ['break_after', 'Break after', 0], ['repeat', 'Repeat after', 0], ['window', 'Open for', 1]]) {
      const cell = gridCell('td'), input = document.createElement('input');
      input.type = 'number'; input.min = min; input.max = 1440; input.required = true;
      input.value = period[key]; input.setAttribute('aria-label', `Period ${index + 1} ${label}`);
      if (isLunch && key === 'break_after') { input.disabled = true; input.title = 'Lunch break: fixed at 60 minutes'; }
      input.addEventListener('input', () => {
        period[key] = input.value === '' ? '' : Number(input.value);
        drawGridHeaders();
      });
      cell.append(input); row.append(cell);
    }
    settings.append(row);
  });
  drawGridHeaders();
  gridDays.forEach((day, dayIndex) => {
    const row = gridCell('tr'); row.append(gridCell('th', day));
    sectionGrid.periods.forEach((_, index) => {
      const cell = gridCell('td'), input = document.createElement('input');
      input.type = 'text'; input.maxLength = 100; input.placeholder = 'Free period';
      input.value = sectionGrid.subjects[dayIndex]?.[index] || '';
      input.setAttribute('aria-label', `${day} Period ${index + 1} subject`);
      input.addEventListener('input', () => { sectionGrid.subjects[dayIndex][index] = input.value; });
      cell.append(input); row.append(cell);
      if (sectionGrid.lunch_after === index + 1) {
        const lunch = gridCell('td', 'Lunch break'); lunch.className = 'lunch-cell';
        lunch.append(gridCell('small', '1 hour · No attendance')); row.append(lunch);
      }
    });
    body.append(row);
  });
}
function drawGridHeaders() {
  const heading = document.getElementById('grid-head');
  const header = gridCell('tr'); header.append(gridCell('th', 'Day / Period'));
  const [hours, minutes] = (sectionGrid.day_start || '09:30').split(':').map(Number);
  let clock = hours * 60 + minutes;
  const formatTime = value => value > 1440 ? 'Past midnight' : `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
  sectionGrid.periods.forEach((period, index) => {
    const cell = gridCell('th', `Period ${index + 1}`);
    cell.append(gridCell('small', `${formatTime(clock)}–${formatTime(clock + Number(period.duration))}`));
    header.append(cell);
    clock += Number(period.duration);
    if (sectionGrid.lunch_after === index + 1) {
      const lunch = gridCell('th', 'Lunch break'); lunch.className = 'lunch-cell';
      lunch.append(gridCell('small', `${formatTime(clock)}–${formatTime(clock + 60)}`)); header.append(lunch);
    }
    clock += Number(period.break_after);
  });
  heading.replaceChildren(header);
}
startField.addEventListener('input', () => { sectionGrid.day_start = startField.value; drawGridHeaders(); });
lunchField.addEventListener('change', () => {
  if (sectionGrid.lunch_after !== null) sectionGrid.periods[sectionGrid.lunch_after - 1].break_after = 0;
  sectionGrid.lunch_after = lunchField.value ? Number(lunchField.value) : null;
  drawGrid();
});
countField.addEventListener('change', () => {
  const count = Number(countField.value), oldCount = sectionGrid.periods.length;
  if (!Number.isInteger(count) || count < 1 || count > 10) return;
  if (count < oldCount && sectionGrid.subjects.some(row => row.slice(count).some(subject => subject.trim())) &&
      !window.confirm('Remove the subjects in the last periods from this draft? Existing attendance history will remain.')) {
    countField.value = oldCount; return;
  }
  while (sectionGrid.periods.length < count) sectionGrid.periods.push({duration: 60, break_after: 0, repeat: 0, window: 10});
  sectionGrid.periods.length = count;
  if (sectionGrid.lunch_after > count) sectionGrid.lunch_after = count;
  sectionGrid.subjects = sectionGrid.subjects.map(row => Array.from({length: count}, (_, i) => row[i] || ''));
  drawGrid();
});
document.getElementById('section-form').addEventListener('submit', () => {
  sectionGrid.enabled = document.getElementById('grid-enabled').checked;
  sectionGrid.day_start = startField.value;
  document.getElementById('grid-json').value = JSON.stringify(sectionGrid);
});
drawGrid();
