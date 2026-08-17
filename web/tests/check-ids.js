/* 开发辅助：校验 JS 中 getElementById('x') 引用的 id 都存在于 index.html */
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const jsFiles = ['../js/live.js', '../js/records.js', '../js/playback.js', '../js/app.js']
  .map(f => fs.readFileSync(path.join(__dirname, f), 'utf8'))
  .join('\n');
const used = [...jsFiles.matchAll(/getElementById\(\s*'([^']+)'\s*\)/g)].map(m => m[1]);
const missing = [...new Set(used.filter(u => !ids.has(u)))];
console.log('HTML ids:', ids.size, '| JS used:', new Set(used).size, '| missing:', missing.length ? missing : 'none');
process.exit(missing.length ? 1 : 0);
