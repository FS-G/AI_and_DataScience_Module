# AI Developer Portfolio — Git, GitHub & Vercel

Build a professional, reusable AI developer portfolio, then publish it to GitHub and deploy it with Vercel.

## Project structure

```text
ai-portfolio/
├── index.html
├── styles.css
├── script.js
└── portfolio.json
```

## `portfolio.json`

All changeable portfolio information belongs in this JSON file. Students can personalize the project without changing the HTML or JavaScript.

```json
{
  "siteTitle": "Alex Morgan | AI Developer",
  "name": "Alex Morgan",
  "initials": "AM",
  "role": "AI Developer & Product Engineer",
  "email": "hello@example.com",
  "intro": "I design and build intelligent products that turn complex data into useful experiences.",
  "about": "I build practical AI products with a focus on generative AI, machine learning, thoughtful UX, and reliable engineering.",
  "github": "https://github.com/your-username",
  "linkedin": "https://linkedin.com/in/your-username",
  "stats": [["04+", "Years building"], ["28", "Projects shipped"], ["12", "AI experiments"]],
  "skills": [["✦", "Generative AI", "Advanced"], ["⌘", "Python & APIs", "Advanced"], ["◈", "Data & ML", "Advanced"], ["◌", "Product Design", "Strong"]],
  "services": [
    ["01", "AI Product Strategy", "Turn a promising AI idea into a clear roadmap, validated use case, and measurable product plan."],
    ["02", "Generative AI Systems", "Build reliable RAG assistants, copilots, evaluation pipelines, and production-ready LLM workflows."],
    ["03", "ML Engineering", "Design data pipelines, forecasting models, APIs, and dashboards that teams can trust and use."],
    ["04", "Technical Workshops", "Help your team understand AI fundamentals, Git, GitHub, deployment, and responsible experimentation."]
  ],
  "process": [["01", "Discover", "Clarify the opportunity, users, constraints, and success metrics."], ["02", "Design", "Shape the experience, technical direction, and smallest useful prototype."], ["03", "Deliver", "Build, test, document, and deploy a reliable product your team can grow."]],
  "testimonial": {"quote": "Alex brought clarity to a complicated AI idea and helped us turn it into a product our customers actually use.", "name": "Maya Chen", "role": "VP Product, Northstar Labs"},
  "projects": [
    {"type":"AI Platform", "name":"Signal Intelligence", "description":"A research workspace that turns documents into cited insights.", "impact":"42% faster research", "tags":["RAG","Python","LLMs"]},
    {"type":"Conversational AI", "name":"CarePath Assistant", "description":"A guided assistant for clear and useful service information.", "impact":"18k monthly conversations", "tags":["NLP","UX","FastAPI"]},
    {"type":"Machine Learning", "name":"Forecast Studio", "description":"A transparent forecasting dashboard for better planning.", "impact":"31% lower forecast error", "tags":["Python","Data","ML"]}
  ]
}
```

## `index.html`

```html
<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Developer Portfolio</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=DM+Mono&family=Manrope:wght@400;600;700;800&display=swap" rel="stylesheet"><link rel="stylesheet" href="styles.css"></head><body>
<header><a class="brand" href="#top"><b id="initials"></b><span id="brand"></span></a><nav><a href="#work">Work</a><a href="#about">About</a><a href="#contact">Contact</a></nav><a id="github" target="_blank">GitHub ↗</a></header>
<main id="top"><section class="hero"><div><p class="eyebrow">AI DEVELOPER / BUILDER / LEARNER</p><h1 id="role"></h1><p id="intro" class="intro"></p><a class="button" href="#work">Explore my work</a><a class="button ghost" href="#contact">Let's connect</a></div><div class="orb">AI</div></section>
<section class="stats" id="stats"></section><section id="work"><p class="eyebrow">01 / SELECTED WORK</p><h2>Intelligence, made useful.</h2><p class="muted">Products at the intersection of intelligence, design, and engineering.</p><div class="projects" id="projects"></div></section>
<section id="services"><p class="eyebrow">02 / SERVICES</p><h2>From first question<br>to real-world impact.</h2><div class="services" id="services-list"></div></section>
<section id="about" class="about"><div><p class="eyebrow">03 / THE APPROACH</p><h2>Building with purpose.</h2><p class="muted" id="about-copy"></p></div><div class="skills" id="skills"></div></section>
<section id="process"><p class="eyebrow">04 / HOW I WORK</p><div class="process" id="process-list"></div><blockquote id="quote"></blockquote><cite id="quote-by"></cite></section>
<section id="contact" class="contact"><p class="eyebrow">05 / START A CONVERSATION</p><h2>Have a meaningful problem to solve?</h2><p class="muted">Tell me what you are building. I am always interested in thoughtful conversations about AI and products.</p><a id="email" class="email"></a></section></main><footer><span>Designed and built with curiosity.</span><span id="footer"></span></footer><script src="script.js"></script></body></html>
```

## `styles.css`

```css
:root{--bg:#090a0e;--panel:#12141b;--text:#f4f4ef;--muted:#9a9ca7;--line:#252832;--lime:#b9f35c;--sans:Manrope,sans-serif;--mono:'DM Mono',monospace}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);background-image:radial-gradient(circle at 80% 10%,#1b2440,transparent 28%)}header,main,footer{width:min(1160px,calc(100% - 64px));margin:auto}header{height:86px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}.brand{color:var(--text);text-decoration:none;font-weight:800;letter-spacing:.08em;display:flex;gap:12px;align-items:center}.brand b{background:var(--lime);color:#111;border-radius:50%;width:32px;height:32px;display:grid;place-items:center;font-size:11px}.brand span{font-size:13px}nav{display:flex;gap:30px}a{color:var(--muted);text-decoration:none;font-size:12px}a:hover{color:var(--lime)}.hero{min-height:600px;display:grid;grid-template-columns:1.1fr .9fr;align-items:center;gap:50px}.eyebrow{color:var(--lime);font:11px var(--mono);letter-spacing:.1em}.hero h1{font-size:clamp(52px,8vw,105px);line-height:.94;letter-spacing:-.08em;margin:25px 0}.intro{color:var(--muted);font-size:17px;line-height:1.8;max-width:470px}.button{display:inline-block;background:var(--lime);color:#111;padding:15px 20px;border-radius:4px;font-weight:800;margin:34px 10px 0 0}.button.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}.orb{justify-self:center;width:245px;height:245px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 30% 25%,#dfff91,#637c3a 30%,#252c27 67%,#0d1110);box-shadow:0 0 100px #b9f35c36,inset -25px -25px 45px #050605;font:32px var(--mono);color:#111}.stats{border-top:1px solid var(--line);border-bottom:1px solid var(--line);display:grid;grid-template-columns:repeat(3,1fr);padding:28px 0;margin-bottom:140px}.stat{padding-left:25px;border-left:1px solid var(--line)}.stat:first-child{border:0;padding-left:0}.stat strong{font-size:35px}.stat small{display:block;color:var(--muted);font:10px var(--mono);margin-top:6px}section{padding-bottom:140px}h2{font-size:clamp(38px,5vw,68px);line-height:1;letter-spacing:-.07em;margin:15px 0}.muted{color:var(--muted);line-height:1.8;max-width:500px}.projects{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:45px}.project{min-height:350px;padding:27px;background:linear-gradient(145deg,var(--panel),#0d0e12);border:1px solid var(--line);display:flex;flex-direction:column;justify-content:space-between;transition:.25s}.project:hover{transform:translateY(-6px);border-color:var(--lime)}.project small{color:var(--lime);font:10px var(--mono)}.project h3{font-size:26px;letter-spacing:-.06em}.project p{font-size:13px;color:var(--muted);line-height:1.7}.tag{display:inline-block;border:1px solid var(--line);padding:6px 8px;margin:4px 4px 0 0;font:10px var(--mono);color:var(--muted)}.impact{color:var(--text)!important}.about{border-top:1px solid var(--line);padding-top:140px}.skills{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:45px}.skill{padding:20px;background:var(--panel);border:1px solid var(--line)}.skill i{font-style:normal;color:var(--lime);font-size:20px}.skill b{display:block;font-size:12px;margin:20px 0 5px}.skill small{color:var(--muted);font:10px var(--mono)}.contact{border-top:1px solid var(--line);padding-top:110px}.email{display:block;color:var(--lime);font-size:22px;margin:26px 0}.contact{padding-bottom:140px}footer{height:80px;border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;color:var(--muted);font:10px var(--mono)}@media(max-width:720px){header,main,footer{width:calc(100% - 36px)}nav{display:none}.hero{grid-template-columns:1fr;padding:90px 0}.orb{width:175px;height:175px}.stats{margin-bottom:90px}.projects,.skills{grid-template-columns:1fr}.about{padding-top:90px}section{padding-bottom:90px}}
```

Add these rules at the end of `styles.css` for the new services and process sections:

```css
#services,#process{padding-top:120px}#services{border-top:1px solid var(--line)}.services{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line);margin-top:45px}.service{background:var(--panel);padding:32px;min-height:190px}.service span{color:var(--lime);font:12px var(--mono)}.service h3{font-size:22px;margin:28px 0 10px}.service p{color:var(--muted);font-size:13px;line-height:1.7}.process{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:45px}.step{border-top:1px solid var(--lime);padding-top:20px}.step strong{font:12px var(--mono);color:var(--lime)}.step h3{font-size:23px}.step p{color:var(--muted);font-size:13px;line-height:1.7}blockquote{max-width:760px;margin:100px 0 18px;padding-left:28px;border-left:2px solid var(--lime);font-size:clamp(22px,3vw,38px);line-height:1.35}cite{color:var(--muted);font:11px var(--mono);font-style:normal}@media(max-width:720px){.services,.process{grid-template-columns:1fr}}
```

## `script.js`

```javascript
async function start(){const p=await fetch('portfolio.json').then(r=>r.json());document.title=p.siteTitle;document.querySelector('#initials').textContent=p.initials;document.querySelector('#brand').textContent=p.name.split(' ')[0].toUpperCase()+'.DEV';document.querySelector('#role').textContent=p.role;document.querySelector('#intro').textContent=p.intro;document.querySelector('#about-copy').textContent=p.about;document.querySelector('#github').href=p.github;document.querySelector('#email').textContent=p.email;document.querySelector('#email').href='mailto:'+p.email;document.querySelector('#footer').textContent='© '+new Date().getFullYear()+' '+p.name;document.querySelector('#stats').innerHTML=p.stats.map(s=>`<div class="stat"><strong>${s[0]}</strong><small>${s[1]}</small></div>`).join('');document.querySelector('#skills').innerHTML=p.skills.map(s=>`<div class="skill"><i>${s[0]}</i><b>${s[1]}</b><small>${s[2]}</small></div>`).join('');document.querySelector('#projects').innerHTML=p.projects.map(x=>`<article class="project"><div><small>${x.type}</small><h3>${x.name}</h3><p>${x.description}</p></div><div><p class="impact">↗ ${x.impact}</p>${x.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div></article>`).join('')}start().catch(console.error);
```

Add these lines inside `start()` before its final closing brace:

```javascript
document.querySelector('#services-list').innerHTML=p.services.map(s=>`<article class="service"><span>${s[0]}</span><h3>${s[1]}</h3><p>${s[2]}</p></article>`).join('');
document.querySelector('#process-list').innerHTML=p.process.map(s=>`<article class="step"><strong>${s[0]}</strong><h3>${s[1]}</h3><p>${s[2]}</p></article>`).join('');
document.querySelector('#quote').textContent='“'+p.testimonial.quote+'”';
document.querySelector('#quote-by').textContent=p.testimonial.name+' · '+p.testimonial.role;
```

## Run locally

Because JavaScript loads JSON with `fetch()`, run a local server:

```bash
python -m http.server 5500
```

Open `http://localhost:5500`.

## Create the Git repository and push to GitHub

Create an empty GitHub repository named `ai-portfolio`, then run:

```bash
git init
git add .
git commit -m "Create AI developer portfolio"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-portfolio.git
git push -u origin main
```

Replace `YOUR-USERNAME` with the GitHub username.

## Deploy with Vercel

1. Visit [vercel.com](https://vercel.com) and sign in with GitHub.
2. Select **Add New → Project**.
3. Import the `ai-portfolio` repository.
4. Keep the framework preset as **Other** and leave the build command empty.
5. Select **Deploy**.

Vercel automatically redeploys the site whenever new commits are pushed to GitHub.

## Update the portfolio

Edit `portfolio.json`, preview locally, then run:

```bash
git add portfolio.json
git commit -m "Update portfolio content"
git push
```
