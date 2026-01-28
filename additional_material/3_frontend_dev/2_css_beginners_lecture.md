# CSS for Beginners - Complete Lecture

## Why Do We Need CSS?

HTML provides the structure and content of a webpage, but it's like a house with no paint, furniture, or decoration. CSS (Cascading Style Sheets) is what makes your website visually appealing and user-friendly by controlling:

- **Appearance** - Colors, fonts, spacing, and layout
- **Responsiveness** - How content adapts to different screen sizes
- **User Experience** - Hover effects, animations, and visual hierarchy
- **Maintainability** - Separating design from content makes updates easier

---

## 1. Getting Started with CSS

### Three Ways to Add CSS

#### 1. Inline CSS (Not Recommended)
```html
<h1 style="color: blue; font-size: 24px;">Hello World</h1>
```

#### 2. Internal CSS
```html
<head>
    <style>
        h1 {
            color: blue;
            font-size: 24px;
        }
    </style>
</head>
```

#### 3. External CSS (Best Practice)
```html
<!-- In your HTML file -->
<head>
    <link rel="stylesheet" href="styles.css">
</head>
```

```css
/* In styles.css file */
h1 {
    color: blue;
    font-size: 24px;
}
```

### 🎨 Complete Example: Colorful Welcome Page

**HTML (welcome.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to CSS!</title>
    <link rel="stylesheet" href="welcome.css">
</head>
<body>
    <div class="welcome-container">
        <h1>🌟 Welcome to CSS Magic! 🌟</h1>
        <p class="intro-text">Watch how CSS transforms plain HTML!</p>
        <div class="color-boxes">
            <div class="box red-box">Red Box</div>
            <div class="box blue-box">Blue Box</div>
            <div class="box green-box">Green Box</div>
        </div>
    </div>
</body>
</html>
```

**CSS (welcome.css):**
```css
/* External CSS - Best Practice! */
body {
    font-family: Arial, sans-serif;
    background: linear-gradient(45deg, #ff9a9e, #fecfef);
    margin: 0;
    padding: 20px;
}

.welcome-container {
    text-align: center;
    max-width: 600px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

h1 {
    color: #ff6b6b;
    font-size: 2.5em;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.intro-text {
    font-size: 1.2em;
    color: #4ecdc4;
    margin-bottom: 30px;
}

.color-boxes {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
}

.box {
    width: 100px;
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: transform 0.3s ease;
}

.box:hover {
    transform: scale(1.1) rotate(5deg);
}

.red-box {
    background-color: #ff6b6b;
}

.blue-box {
    background-color: #4ecdc4;
}

.green-box {
    background-color: #45b7d1;
}
```

---

## 2. CSS Syntax

### Basic Structure
```css
selector {
    property: value;
    property: value;
}
```

### Example
```css
h1 {
    color: red;
    font-size: 32px;
    text-align: center;
}
```

- **Selector** - Targets HTML elements
- **Property** - What you want to change
- **Value** - How you want to change it
- **Declaration** - Property + value pair
- **Rule** - Complete selector + declarations

### 🎯 Complete Example: Syntax Playground

**HTML (syntax.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Syntax Playground</title>
    <link rel="stylesheet" href="syntax.css">
</head>
<body>
    <div class="playground">
        <h1 class="main-title">CSS Syntax Rules!</h1>
        <p class="description">Each rule follows: selector { property: value; }</p>
        
        <div class="demo-section">
            <h2 class="section-title">Text Styling</h2>
            <p class="styled-text">This text has multiple CSS properties applied!</p>
        </div>
        
        <div class="demo-section">
            <h2 class="section-title">Box Styling</h2>
            <div class="styled-box">I'm a styled box!</div>
        </div>
    </div>
</body>
</html>
```

**CSS (syntax.css):**
```css
/* CSS Syntax Examples */

/* Selector: body */
/* Properties: background, font-family, margin */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
}

/* Selector: .playground */
/* Properties: max-width, margin, background, padding, border-radius */
.playground {
    max-width: 800px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
}

/* Selector: .main-title */
/* Properties: color, font-size, text-align, margin-bottom */
.main-title {
    color: #2c3e50;
    font-size: 2.5em;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

/* Selector: .description */
/* Properties: color, font-size, text-align, font-style */
.description {
    color: #7f8c8d;
    font-size: 1.1em;
    text-align: center;
    font-style: italic;
    margin-bottom: 40px;
}

/* Selector: .demo-section */
/* Properties: margin-bottom, padding, background, border-radius */
.demo-section {
    margin-bottom: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 10px;
    border-left: 4px solid #3498db;
}

/* Selector: .section-title */
/* Properties: color, font-size, margin-bottom */
.section-title {
    color: #2980b9;
    font-size: 1.5em;
    margin-bottom: 15px;
}

/* Selector: .styled-text */
/* Properties: color, font-size, line-height, font-weight */
.styled-text {
    color: #e74c3c;
    font-size: 1.2em;
    line-height: 1.6;
    font-weight: 600;
}

/* Selector: .styled-box */
/* Properties: width, height, background, color, display, align-items, justify-content, border-radius, margin */
.styled-box {
    width: 200px;
    height: 100px;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    margin: 10px 0;
    font-weight: bold;
    font-size: 1.1em;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}

.styled-box:hover {
    transform: translateY(-5px) scale(1.05);
}
```

---

## 3. CSS Selectors

### Element Selectors
```css
/* Selects all paragraphs */
p {
    color: gray;
}

/* Selects all headings level 2 */
h2 {
    font-weight: bold;
}
```

### Class Selectors
```html
<p class="highlight">This text is highlighted</p>
<div class="container">Content here</div>
```

```css
.highlight {
    background-color: yellow;
}

.container {
    max-width: 800px;
    margin: 0 auto;
}
```

### ID Selectors
```html
<header id="main-header">Site Header</header>
```

```css
#main-header {
    background-color: navy;
    color: white;
}
```

### Descendant Selectors
```css
/* Selects paragraphs inside divs */
div p {
    line-height: 1.6;
}

/* Selects links inside navigation */
nav a {
    text-decoration: none;
}
```

### 🎪 Complete Example: Selector Circus

**HTML (selectors.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Selector Circus</title>
    <link rel="stylesheet" href="selectors.css">
</head>
<body>
    <header id="main-header">
        <h1>🎪 CSS Selector Circus 🎪</h1>
        <p>Watch different selectors in action!</p>
    </header>
    
    <nav class="navigation">
        <a href="#" class="nav-link">Home</a>
        <a href="#" class="nav-link">About</a>
        <a href="#" class="nav-link">Contact</a>
    </nav>
    
    <main class="container">
        <section class="highlight">
            <h2>Element Selector Demo</h2>
            <p>This paragraph is styled with element selector!</p>
            <p>So is this one!</p>
        </section>
        
        <section class="special-section">
            <h2>Class Selector Demo</h2>
            <div class="highlight">
                <p>This div has the "highlight" class!</p>
            </div>
            <div class="container">
                <p>This div has the "container" class!</p>
            </div>
        </section>
        
        <section class="descendant-demo">
            <h2>Descendant Selector Demo</h2>
            <div class="parent-box">
                <p>I'm a paragraph inside a div!</p>
                <span>I'm a span inside the same div!</span>
            </div>
            <p>I'm a paragraph outside the div!</p>
        </section>
    </main>
</body>
</html>
```

**CSS (selectors.css):**
```css
/* Element Selectors - Target all elements of that type */
body {
    font-family: 'Comic Sans MS', cursive, sans-serif;
    background: linear-gradient(45deg, #ff9a9e, #fad0c4);
    margin: 0;
    padding: 20px;
}

h1 {
    color: #ff6b6b;
    text-align: center;
    font-size: 2.5em;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
}

h2 {
    color: #4ecdc4;
    font-size: 1.8em;
    margin-bottom: 15px;
    border-bottom: 3px solid #4ecdc4;
    padding-bottom: 5px;
}

p {
    color: #2c3e50;
    font-size: 1.1em;
    line-height: 1.6;
    margin-bottom: 10px;
}

/* ID Selector - Targets element with specific ID */
#main-header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

/* Class Selectors - Target elements with specific class */
.highlight {
    background-color: #ffeb3b;
    padding: 15px;
    border-radius: 10px;
    border: 3px dashed #ff9800;
    margin: 10px 0;
    animation: pulse 2s infinite;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.navigation {
    text-align: center;
    margin-bottom: 30px;
}

.nav-link {
    display: inline-block;
    background: #4ecdc4;
    color: white;
    padding: 10px 20px;
    margin: 0 10px;
    text-decoration: none;
    border-radius: 25px;
    transition: all 0.3s ease;
    font-weight: bold;
}

.nav-link:hover {
    background: #45b7d1;
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

/* Descendant Selectors - Target elements inside other elements */
.navigation a {
    text-decoration: none;
}

.parent-box {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}

.parent-box p {
    color: white;
    font-weight: bold;
    background: rgba(255, 255, 255, 0.2);
    padding: 10px;
    border-radius: 5px;
}

.special-section {
    margin: 30px 0;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 10px;
    border-left: 5px solid #ff6b6b;
}

.descendant-demo {
    margin: 30px 0;
    padding: 20px;
    background: #e8f5e8;
    border-radius: 10px;
    border-left: 5px solid #4ecdc4;
}

/* Animation for highlight class */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
```

---

## 4. Colors in CSS

### Color Formats
```css
.examples {
    /* Named colors */
    color: red;
    
    /* Hexadecimal */
    background-color: #ff0000;
    
    /* RGB */
    border-color: rgb(255, 0, 0);
    
    /* RGBA (with transparency) */
    box-shadow: rgba(0, 0, 0, 0.5);
}
```

### Common Color Properties
```css
.color-demo {
    color: blue;              /* Text color */
    background-color: lightblue;  /* Background color */
    border-color: darkblue;   /* Border color */
}
```

### 🌈 Complete Example: Color Palette Explorer

**HTML (colors.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Color Palette Explorer</title>
    <link rel="stylesheet" href="colors.css">
</head>
<body>
    <div class="color-showcase">
        <h1>🌈 Color Palette Explorer 🌈</h1>
        <p class="subtitle">Discover different ways to use colors in CSS!</p>
        
        <div class="color-section">
            <h2>Named Colors</h2>
            <div class="color-grid">
                <div class="color-box red">Red</div>
                <div class="color-box blue">Blue</div>
                <div class="color-box green">Green</div>
                <div class="color-box purple">Purple</div>
            </div>
        </div>
        
        <div class="color-section">
            <h2>Hexadecimal Colors</h2>
            <div class="color-grid">
                <div class="color-box hex-1">#FF6B6B</div>
                <div class="color-box hex-2">#4ECDC4</div>
                <div class="color-box hex-3">#45B7D1</div>
                <div class="color-box hex-4">#96CEB4</div>
            </div>
        </div>
        
        <div class="color-section">
            <h2>RGB Colors</h2>
            <div class="color-grid">
                <div class="color-box rgb-1">RGB(255, 107, 107)</div>
                <div class="color-box rgb-2">RGB(78, 205, 196)</div>
                <div class="color-box rgb-3">RGB(69, 183, 209)</div>
                <div class="color-box rgb-4">RGB(150, 206, 180)</div>
            </div>
        </div>
        
        <div class="color-section">
            <h2>RGBA Colors (with Transparency)</h2>
            <div class="color-grid">
                <div class="color-box rgba-1">RGBA(255, 107, 107, 0.8)</div>
                <div class="color-box rgba-2">RGBA(78, 205, 196, 0.6)</div>
                <div class="color-box rgba-3">RGBA(69, 183, 209, 0.4)</div>
                <div class="color-box rgba-4">RGBA(150, 206, 180, 0.2)</div>
            </div>
        </div>
        
        <div class="gradient-section">
            <h2>Gradient Backgrounds</h2>
            <div class="gradient-box gradient-1">Linear Gradient</div>
            <div class="gradient-box gradient-2">Radial Gradient</div>
        </div>
    </div>
</body>
</html>
```

**CSS (colors.css):**
```css
/* Color Showcase Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.color-showcase {
    max-width: 1000px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

.color-section {
    margin-bottom: 40px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

.color-section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.5em;
}

.color-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.color-box {
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: all 0.3s ease;
    cursor: pointer;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.color-box:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

/* Named Colors */
.red {
    background-color: red;
}

.blue {
    background-color: blue;
}

.green {
    background-color: green;
}

.purple {
    background-color: purple;
}

/* Hexadecimal Colors */
.hex-1 {
    background-color: #FF6B6B;
}

.hex-2 {
    background-color: #4ECDC4;
}

.hex-3 {
    background-color: #45B7D1;
}

.hex-4 {
    background-color: #96CEB4;
}

/* RGB Colors */
.rgb-1 {
    background-color: rgb(255, 107, 107);
}

.rgb-2 {
    background-color: rgb(78, 205, 196);
}

.rgb-3 {
    background-color: rgb(69, 183, 209);
}

.rgb-4 {
    background-color: rgb(150, 206, 180);
}

/* RGBA Colors (with transparency) */
.rgba-1 {
    background-color: rgba(255, 107, 107, 0.8);
}

.rgba-2 {
    background-color: rgba(78, 205, 196, 0.6);
}

.rgba-3 {
    background-color: rgba(69, 183, 209, 0.4);
}

.rgba-4 {
    background-color: rgba(150, 206, 180, 0.2);
}

/* Gradient Section */
.gradient-section {
    margin-bottom: 40px;
    padding: 20px;
    background: #e8f5e8;
    border-radius: 15px;
    border-left: 5px solid #27ae60;
}

.gradient-box {
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    border-radius: 15px;
    margin: 15px 0;
    font-size: 1.2em;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
}

.gradient-box:hover {
    transform: translateY(-5px);
}

.gradient-1 {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
}

.gradient-2 {
    background: radial-gradient(circle, #ff6b6b, #4ecdc4, #45b7d1);
}
```

---

## 5. Typography

### Font Properties
```css
.text-styling {
    font-family: Arial, sans-serif;  /* Font family */
    font-size: 18px;                 /* Size */
    font-weight: bold;               /* Weight */
    font-style: italic;              /* Style */
    text-align: center;              /* Alignment */
    text-decoration: underline;      /* Decoration */
    line-height: 1.5;                /* Line spacing */
    letter-spacing: 2px;             /* Character spacing */
}
```

### Web Safe Font Stacks
```css
.font-examples {
    /* Serif fonts */
    font-family: "Times New Roman", Times, serif;
    
    /* Sans-serif fonts */
    font-family: Arial, Helvetica, sans-serif;
    
    /* Monospace fonts */
    font-family: "Courier New", Courier, monospace;
}
```

### 📝 Complete Example: Typography Playground

**HTML (typography.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Typography Playground</title>
    <link rel="stylesheet" href="typography.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Dancing+Script:wght@400;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="typography-showcase">
        <h1>📝 Typography Playground 📝</h1>
        <p class="intro">Explore the power of fonts and text styling!</p>
        
        <section class="font-family-demo">
            <h2>Font Families</h2>
            <div class="font-sample serif-font">
                <h3>Serif Font</h3>
                <p>"Times New Roman" - Perfect for formal documents</p>
            </div>
            <div class="font-sample sans-serif-font">
                <h3>Sans-Serif Font</h3>
                <p>Arial - Clean and modern for web content</p>
            </div>
            <div class="font-sample monospace-font">
                <h3>Monospace Font</h3>
                <p>"Courier New" - Great for code and data</p>
            </div>
            <div class="font-sample google-font">
                <h3>Google Font</h3>
                <p>Roboto - Beautiful and versatile</p>
            </div>
        </section>
        
        <section class="font-size-demo">
            <h2>Font Sizes</h2>
            <div class="size-examples">
                <p class="size-small">Small text (12px)</p>
                <p class="size-medium">Medium text (16px)</p>
                <p class="size-large">Large text (24px)</p>
                <p class="size-xlarge">Extra Large text (32px)</p>
            </div>
        </section>
        
        <section class="font-weight-demo">
            <h2>Font Weights</h2>
            <div class="weight-examples">
                <p class="weight-light">Light weight (300)</p>
                <p class="weight-normal">Normal weight (400)</p>
                <p class="weight-bold">Bold weight (700)</p>
                <p class="weight-bolder">Bolder weight (900)</p>
            </div>
        </section>
        
        <section class="text-decoration-demo">
            <h2>Text Decorations</h2>
            <div class="decoration-examples">
                <p class="underline">Underlined text</p>
                <p class="overline">Overlined text</p>
                <p class="line-through">Strikethrough text</p>
                <p class="no-decoration">No decoration</p>
            </div>
        </section>
        
        <section class="text-align-demo">
            <h2>Text Alignment</h2>
            <div class="align-examples">
                <p class="align-left">Left aligned text - Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                <p class="align-center">Center aligned text - Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                <p class="align-right">Right aligned text - Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                <p class="align-justify">Justified text - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            </div>
        </section>
        
        <section class="special-effects">
            <h2>Special Text Effects</h2>
            <div class="effects-examples">
                <p class="text-shadow">Text with shadow effect</p>
                <p class="letter-spacing">Letter spacing example</p>
                <p class="word-spacing">Word spacing example</p>
                <p class="line-height">Line height example - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            </div>
        </section>
    </div>
</body>
</html>
```

**CSS (typography.css):**
```css
/* Typography Showcase Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    line-height: 1.6;
}

.typography-showcase {
    max-width: 1000px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

section {
    margin-bottom: 40px;
    padding: 25px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

/* Font Family Demo */
.font-sample {
    margin-bottom: 20px;
    padding: 15px;
    background: white;
    border-radius: 10px;
    border: 2px solid #e9ecef;
}

.font-sample h3 {
    color: #495057;
    margin-bottom: 10px;
    font-size: 1.2em;
}

.serif-font {
    font-family: "Times New Roman", Times, serif;
}

.sans-serif-font {
    font-family: Arial, Helvetica, sans-serif;
}

.monospace-font {
    font-family: "Courier New", Courier, monospace;
}

.google-font {
    font-family: 'Roboto', sans-serif;
}

/* Font Size Demo */
.size-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.size-small {
    font-size: 12px;
    color: #6c757d;
}

.size-medium {
    font-size: 16px;
    color: #495057;
}

.size-large {
    font-size: 24px;
    color: #343a40;
}

.size-xlarge {
    font-size: 32px;
    color: #212529;
    font-weight: bold;
}

/* Font Weight Demo */
.weight-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.weight-light {
    font-weight: 300;
    color: #6c757d;
}

.weight-normal {
    font-weight: 400;
    color: #495057;
}

.weight-bold {
    font-weight: 700;
    color: #343a40;
}

.weight-bolder {
    font-weight: 900;
    color: #212529;
}

/* Text Decoration Demo */
.decoration-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.underline {
    text-decoration: underline;
    color: #dc3545;
}

.overline {
    text-decoration: overline;
    color: #fd7e14;
}

.line-through {
    text-decoration: line-through;
    color: #6f42c1;
}

.no-decoration {
    text-decoration: none;
    color: #28a745;
}

/* Text Alignment Demo */
.align-examples {
    display: grid;
    gap: 20px;
}

.align-left {
    text-align: left;
    background: #e3f2fd;
    padding: 15px;
    border-radius: 8px;
}

.align-center {
    text-align: center;
    background: #f3e5f5;
    padding: 15px;
    border-radius: 8px;
}

.align-right {
    text-align: right;
    background: #e8f5e8;
    padding: 15px;
    border-radius: 8px;
}

.align-justify {
    text-align: justify;
    background: #fff3e0;
    padding: 15px;
    border-radius: 8px;
}

/* Special Effects Demo */
.effects-examples {
    display: grid;
    gap: 20px;
}

.text-shadow {
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    font-size: 1.5em;
    color: #2c3e50;
    font-weight: bold;
}

.letter-spacing {
    letter-spacing: 3px;
    font-size: 1.2em;
    color: #e74c3c;
    font-weight: bold;
}

.word-spacing {
    word-spacing: 10px;
    font-size: 1.2em;
    color: #3498db;
    font-weight: bold;
}

.line-height {
    line-height: 2.5;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    color: #495057;
}
```

---

## 6. The Box Model

Every HTML element is a rectangular box with four parts:

```css
.box-model-demo {
    width: 200px;           /* Content width */
    height: 100px;          /* Content height */
    padding: 20px;          /* Space inside the border */
    border: 2px solid black; /* Border around the element */
    margin: 10px;           /* Space outside the border */
}
```

### Box Model Visualization
```
┌─────────────────────────────────────┐
│              MARGIN                 │
│  ┌───────────────────────────────┐  │
│  │           BORDER              │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │        PADDING          │  │  │
│  │  │  ┌─────────────────┐    │  │  │
│  │  │  │    CONTENT      │    │  │  │
│  │  │  └─────────────────┘    │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Margin and Padding Shortcuts
```css
.spacing-examples {
    /* All sides */
    margin: 10px;
    padding: 15px;
    
    /* Top/bottom, left/right */
    margin: 10px 20px;
    padding: 15px 25px;
    
    /* Top, right, bottom, left */
    margin: 10px 20px 30px 40px;
    
    /* Individual sides */
    margin-top: 10px;
    padding-left: 20px;
}
```

### 📦 Complete Example: Box Model Visualizer

**HTML (boxmodel.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Box Model Visualizer</title>
    <link rel="stylesheet" href="boxmodel.css">
</head>
<body>
    <div class="box-model-showcase">
        <h1>📦 Box Model Visualizer 📦</h1>
        <p class="intro">See how margin, border, padding, and content work together!</p>
        
        <section class="basic-box-demo">
            <h2>Basic Box Model</h2>
            <div class="box-container">
                <div class="basic-box">
                    <div class="content">CONTENT</div>
                </div>
            </div>
            <div class="box-info">
                <p><strong>Content:</strong> 200px × 100px</p>
                <p><strong>Padding:</strong> 20px all sides</p>
                <p><strong>Border:</strong> 3px solid</p>
                <p><strong>Margin:</strong> 15px all sides</p>
            </div>
        </section>
        
        <section class="spacing-demo">
            <h2>Margin & Padding Variations</h2>
            <div class="spacing-examples">
                <div class="spacing-box uniform">
                    <div class="content">Uniform<br>10px all</div>
                </div>
                <div class="spacing-box vertical-horizontal">
                    <div class="content">Vertical/Horizontal<br>10px 20px</div>
                </div>
                <div class="spacing-box four-sides">
                    <div class="content">Four Sides<br>10px 20px 30px 40px</div>
                </div>
                <div class="spacing-box individual">
                    <div class="content">Individual<br>top, right, bottom, left</div>
                </div>
            </div>
        </section>
        
        <section class="box-sizing-demo">
            <h2>Box Sizing Property</h2>
            <div class="sizing-comparison">
                <div class="content-box">
                    <h3>content-box (default)</h3>
                    <div class="demo-box content-sizing">
                        <div class="inner-content">Width: 200px + padding + border</div>
                    </div>
                </div>
                <div class="border-box">
                    <h3>border-box</h3>
                    <div class="demo-box border-sizing">
                        <div class="inner-content">Width: 200px total</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="interactive-demo">
            <h2>Interactive Box Model</h2>
            <div class="interactive-container">
                <div class="interactive-box">
                    <div class="margin-layer">
                        <div class="border-layer">
                            <div class="padding-layer">
                                <div class="content-layer">
                                    Hover me!
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="controls">
                <button onclick="toggleMargin()">Toggle Margin</button>
                <button onclick="toggleBorder()">Toggle Border</button>
                <button onclick="togglePadding()">Toggle Padding</button>
            </div>
        </section>
    </div>
    
    <script>
        function toggleMargin() {
            document.querySelector('.interactive-box').classList.toggle('show-margin');
        }
        function toggleBorder() {
            document.querySelector('.interactive-box').classList.toggle('show-border');
        }
        function togglePadding() {
            document.querySelector('.interactive-box').classList.toggle('show-padding');
        }
    </script>
</body>
</html>
```

**CSS (boxmodel.css):**
```css
/* Box Model Visualizer Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.box-model-showcase {
    max-width: 1000px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

section {
    margin-bottom: 40px;
    padding: 25px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

/* Basic Box Demo */
.box-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.basic-box {
    width: 200px;
    height: 100px;
    padding: 20px;
    border: 3px solid #e74c3c;
    margin: 15px;
    background: #ecf0f1;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.content {
    width: 100%;
    height: 100%;
    background: #3498db;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.1em;
    border-radius: 5px;
}

.box-info {
    text-align: center;
    background: white;
    padding: 15px;
    border-radius: 10px;
    border: 2px solid #e9ecef;
}

.box-info p {
    margin: 5px 0;
    color: #495057;
}

/* Spacing Demo */
.spacing-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.spacing-box {
    background: #e8f5e8;
    border: 2px solid #27ae60;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}

.uniform {
    margin: 10px;
    padding: 10px;
}

.vertical-horizontal {
    margin: 10px 20px;
    padding: 10px 20px;
}

.four-sides {
    margin: 10px 20px 30px 40px;
    padding: 10px 20px 30px 40px;
}

.individual {
    margin-top: 10px;
    margin-right: 20px;
    margin-bottom: 30px;
    margin-left: 40px;
    padding-top: 10px;
    padding-right: 20px;
    padding-bottom: 30px;
    padding-left: 40px;
}

/* Box Sizing Demo */
.sizing-comparison {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
}

.content-box, .border-box {
    text-align: center;
}

.content-box h3, .border-box h3 {
    color: #2c3e50;
    margin-bottom: 15px;
}

.demo-box {
    width: 200px;
    height: 100px;
    padding: 20px;
    border: 3px solid #e74c3c;
    margin: 0 auto;
    background: #ecf0f1;
    border-radius: 10px;
}

.content-sizing {
    box-sizing: content-box; /* Default */
}

.border-sizing {
    box-sizing: border-box;
}

.inner-content {
    width: 100%;
    height: 100%;
    background: #9b59b6;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    border-radius: 5px;
    font-size: 0.9em;
    text-align: center;
}

/* Interactive Demo */
.interactive-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.interactive-box {
    position: relative;
    transition: all 0.3s ease;
}

.margin-layer {
    background: rgba(255, 193, 7, 0.3);
    padding: 20px;
    border-radius: 15px;
    transition: all 0.3s ease;
}

.border-layer {
    background: rgba(220, 53, 69, 0.3);
    padding: 15px;
    border-radius: 10px;
    transition: all 0.3s ease;
}

.padding-layer {
    background: rgba(40, 167, 69, 0.3);
    padding: 15px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.content-layer {
    background: #007bff;
    color: white;
    padding: 20px;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
    font-size: 1.1em;
    transition: all 0.3s ease;
}

.interactive-box:hover .content-layer {
    background: #0056b3;
    transform: scale(1.05);
}

.show-margin .margin-layer {
    background: rgba(255, 193, 7, 0.8);
    border: 2px dashed #ffc107;
}

.show-border .border-layer {
    background: rgba(220, 53, 69, 0.8);
    border: 2px solid #dc3545;
}

.show-padding .padding-layer {
    background: rgba(40, 167, 69, 0.8);
    border: 2px solid #28a745;
}

.controls {
    text-align: center;
}

.controls button {
    background: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    margin: 5px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    transition: background 0.3s ease;
}

.controls button:hover {
    background: #0056b3;
}
```

---

## 7. Display Property

### Block Elements
```css
.block-element {
    display: block;  /* Takes full width, starts on new line */
    width: 100%;
    background-color: lightgray;
}
```

### Inline Elements
```css
.inline-element {
    display: inline;  /* Only takes necessary width, stays in line */
    background-color: yellow;
}
```

### Inline-Block Elements
```css
.inline-block-element {
    display: inline-block;  /* Combines benefits of both */
    width: 150px;
    height: 50px;
    background-color: lightblue;
}
```

### 🎭 Complete Example: Display Property Theater

**HTML (display.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Display Property Theater</title>
    <link rel="stylesheet" href="display.css">
</head>
<body>
    <div class="display-showcase">
        <h1>🎭 Display Property Theater 🎭</h1>
        <p class="intro">Watch how different display values change element behavior!</p>
        
        <section class="block-demo">
            <h2>Block Elements</h2>
            <p class="demo-description">Block elements take full width and start on new lines</p>
            <div class="block-container">
                <div class="block-box box-1">Block Box 1</div>
                <div class="block-box box-2">Block Box 2</div>
                <div class="block-box box-3">Block Box 3</div>
            </div>
        </section>
        
        <section class="inline-demo">
            <h2>Inline Elements</h2>
            <p class="demo-description">Inline elements only take necessary width and stay in line</p>
            <div class="inline-container">
                <span class="inline-box box-1">Inline Box 1</span>
                <span class="inline-box box-2">Inline Box 2</span>
                <span class="inline-box box-3">Inline Box 3</span>
            </div>
        </section>
        
        <section class="inline-block-demo">
            <h2>Inline-Block Elements</h2>
            <p class="demo-description">Inline-block combines benefits of both - width control + inline behavior</p>
            <div class="inline-block-container">
                <div class="inline-block-box box-1">Inline-Block Box 1</div>
                <div class="inline-block-box box-2">Inline-Block Box 2</div>
                <div class="inline-block-box box-3">Inline-Block Box 3</div>
            </div>
        </section>
        
        <section class="none-demo">
            <h2>Display: None</h2>
            <p class="demo-description">Elements with display: none are completely hidden</p>
            <div class="none-container">
                <div class="visible-box">I'm visible!</div>
                <div class="hidden-box">I'm hidden!</div>
                <div class="visible-box">I'm also visible!</div>
            </div>
            <button onclick="toggleHidden()" class="toggle-btn">Toggle Hidden Box</button>
        </section>
        
        <section class="flex-demo">
            <h2>Display: Flex (Preview)</h2>
            <p class="demo-description">Flex creates a flexible container for layout</p>
            <div class="flex-container">
                <div class="flex-item">Flex Item 1</div>
                <div class="flex-item">Flex Item 2</div>
                <div class="flex-item">Flex Item 3</div>
            </div>
        </section>
    </div>
    
    <script>
        function toggleHidden() {
            const hiddenBox = document.querySelector('.hidden-box');
            hiddenBox.style.display = hiddenBox.style.display === 'none' ? 'block' : 'none';
        }
    </script>
</body>
</html>
```

**CSS (display.css):**
```css
/* Display Property Theater Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.display-showcase {
    max-width: 1000px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

section {
    margin-bottom: 40px;
    padding: 25px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

section h2 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

.demo-description {
    color: #6c757d;
    margin-bottom: 20px;
    font-style: italic;
}

/* Block Elements Demo */
.block-container {
    background: #e3f2fd;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #2196f3;
}

.block-box {
    display: block;
    background: #2196f3;
    color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    transition: all 0.3s ease;
}

.block-box:hover {
    background: #1976d2;
    transform: translateX(10px);
}

/* Inline Elements Demo */
.inline-container {
    background: #f3e5f5;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #9c27b0;
}

.inline-box {
    display: inline;
    background: #9c27b0;
    color: white;
    padding: 8px 12px;
    margin: 5px;
    border-radius: 5px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.inline-box:hover {
    background: #7b1fa2;
    transform: scale(1.1);
}

/* Inline-Block Elements Demo */
.inline-block-container {
    background: #e8f5e8;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #4caf50;
}

.inline-block-box {
    display: inline-block;
    width: 200px;
    height: 80px;
    background: #4caf50;
    color: white;
    padding: 15px;
    margin: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    line-height: 50px;
    transition: all 0.3s ease;
}

.inline-block-box:hover {
    background: #388e3c;
    transform: rotate(5deg) scale(1.05);
}

/* Display: None Demo */
.none-container {
    background: #fff3e0;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #ff9800;
}

.visible-box {
    display: block;
    background: #ff9800;
    color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

.hidden-box {
    display: none;
    background: #f44336;
    color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

.toggle-btn {
    background: #ff9800;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    margin-top: 15px;
    transition: background 0.3s ease;
}

.toggle-btn:hover {
    background: #f57c00;
}

/* Flex Demo (Preview) */
.flex-container {
    display: flex;
    background: #e1f5fe;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #00bcd4;
    gap: 15px;
    justify-content: space-around;
}

.flex-item {
    background: #00bcd4;
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    flex: 1;
    transition: all 0.3s ease;
}

.flex-item:hover {
    background: #0097a7;
    transform: translateY(-5px);
}
```

---

## 8. Positioning

CSS positioning controls where elements are placed on the page. There are 5 main positioning values:

### 1. Static (Default)
```css
.static-box {
    position: static;  /* Normal document flow - default behavior */
}
```
- Elements follow normal document flow
- `top`, `right`, `bottom`, `left` properties have no effect

### 2. Relative
```css
.relative-box {
    position: relative;
    top: 20px;    /* Moves 20px down from original position */
    left: 30px;   /* Moves 30px right from original position */
}
```
- Element stays in normal flow but can be moved from its original position
- Other elements don't move to fill the space

### 3. Absolute
```css
.absolute-box {
    position: absolute;
    top: 50px;    /* 50px from top of nearest positioned parent */
    right: 20px;  /* 20px from right edge */
}
```
- Element is removed from normal flow
- **Looks for the nearest parent with positioning** (`relative`, `absolute`, `fixed`, or `sticky`)
- If no positioned parent exists, positions relative to viewport
- Other elements act like it doesn't exist

### 4. Fixed
```css
.fixed-box {
    position: fixed;
    bottom: 20px;  /* Always 20px from bottom of viewport */
    right: 20px;   /* Always 20px from right of viewport */
}
```
- Element is positioned relative to viewport
- Stays in same place when scrolling
- Removed from normal flow

### 5. Sticky
```css
.sticky-box {
    position: sticky;
    top: 10px;     /* Sticks 10px from top when scrolling */
}
```
- Switches between relative and fixed based on scroll position
- Sticks to specified position when scrolling

### 📦 Complete Example: Simple Positioning Demo

**HTML (positioning.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Positioning Demo</title>
    <link rel="stylesheet" href="positioning.css">
</head>
<body>
    <div class="container">
        <h1>CSS Positioning Demo</h1>
        
        <div class="demo-section">
            <h2>1. Static (Default)</h2>
            <div class="box static-box">Static Box</div>
            <div class="box static-box">Another Static Box</div>
        </div>
        
        <div class="demo-section">
            <h2>2. Relative</h2>
            <div class="box relative-box">Relative Box</div>
            <div class="box static-box">Normal Box After</div>
        </div>
        
        <div class="demo-section">
            <h2>3. Absolute</h2>
            <h3>With Positioned Parent (relative):</h3>
            <div class="absolute-container">
                <div class="box absolute-box">Absolute Box</div>
                <div class="box static-box">Normal Box</div>
            </div>
            
            <h3>Without Positioned Parent:</h3>
            <div class="no-position-container">
                <div class="box absolute-no-parent">Absolute Box (no positioned parent)</div>
                <div class="box static-box">Normal Box</div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>4. Fixed</h2>
            <div class="box fixed-box">Fixed Box</div>
            <p>Scroll down to see the fixed box stay in place!</p>
        </div>
        
        <div class="demo-section">
            <h2>5. Sticky</h2>
            <div class="box sticky-box">Sticky Box</div>
            <div class="long-content">
                <p>Scroll down to see sticky behavior...</p>
                <p>More content...</p>
                <p>Keep scrolling...</p>
                <p>Almost there...</p>
                <p>Last content!</p>
            </div>
        </div>
    </div>
</body>
</html>
```

**CSS (positioning.css):**
```css
/* Simple Positioning Demo */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    line-height: 1.6;
}

.container {
    max-width: 800px;
    margin: 0 auto;
}

h1 {
    text-align: center;
    color: #333;
    margin-bottom: 30px;
}

.demo-section {
    margin-bottom: 40px;
    padding: 20px;
    border: 2px solid #ddd;
    border-radius: 8px;
    background: #f9f9f9;
}

.demo-section h2 {
    color: #555;
    margin-bottom: 15px;
}

/* Basic box styling */
.box {
    width: 150px;
    height: 80px;
    background: #4CAF50;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 10px 0;
    border-radius: 5px;
    font-weight: bold;
}

/* 1. Static positioning (default) */
.static-box {
    position: static;
    background: #2196F3;
}

/* 2. Relative positioning */
.relative-box {
    position: relative;
    top: 20px;
    left: 30px;
    background: #FF9800;
}

/* 3. Absolute positioning */
.absolute-container {
    position: relative;  /* This makes it a positioned parent */
    height: 200px;
    background: #E3F2FD;
    border: 2px dashed #2196F3;
    padding: 20px;
}

.absolute-box {
    position: absolute;
    top: 50px;
    right: 50px;
    background: #9C27B0;
}

.no-position-container {
    height: 200px;
    background: #FFF3E0;
    border: 2px dashed #FF9800;
    padding: 20px;
    margin-top: 20px;
}

.absolute-no-parent {
    position: absolute;
    top: 50px;
    right: 50px;
    background: #E91E63;
}

/* 4. Fixed positioning */
.fixed-box {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #F44336;
    z-index: 1000;
}

/* 5. Sticky positioning */
.sticky-box {
    position: sticky;
    top: 20px;
    background: #607D8B;
}

.long-content {
    height: 300px;
    overflow-y: auto;
    background: white;
    padding: 20px;
    border: 1px solid #ccc;
}

.long-content p {
    margin: 20px 0;
    padding: 10px;
    background: #f0f0f0;
}
```

---

## 9. Flexbox Basics

**Why Flexbox?** Before flexbox, centering elements vertically was a nightmare! You had to use complex tricks like `position: absolute` with transforms, or table-cell displays. Flexbox makes it simple to:
- Center content both horizontally and vertically
- Distribute space evenly between items
- Create responsive layouts that adapt to different screen sizes
- Align items in rows or columns with ease

### Flex Container
```css
.flex-container {
    display: flex;
    justify-content: space-between;  /* Horizontal alignment */
    align-items: center;             /* Vertical alignment */
    gap: 20px;                      /* Space between items */
}
```

### Complete Flexbox Properties

#### Container Properties (Parent)
```css
.flex-container {
    display: flex;                    /* flex, inline-flex */
    
    /* Direction */
    flex-direction: row;             /* row, row-reverse, column, column-reverse */
    
    /* Wrapping */
    flex-wrap: nowrap;              /* nowrap, wrap, wrap-reverse */
    
    /* Main axis alignment (horizontal if row, vertical if column) */
    justify-content: flex-start;    /* flex-start, flex-end, center, space-between, space-around, space-evenly */
    
    /* Cross axis alignment (vertical if row, horizontal if column) */
    align-items: stretch;           /* flex-start, flex-end, center, stretch, baseline */
    
    /* Multiple lines alignment */
    align-content: stretch;         /* flex-start, flex-end, center, stretch, space-between, space-around, space-evenly */
    
    /* Gap between items */
    gap: 10px;                      /* Any length value */
    row-gap: 10px;                  /* Gap between rows */
    column-gap: 10px;               /* Gap between columns */
}
```

#### Item Properties (Children)
```css
.flex-item {
    /* Growth factor */
    flex-grow: 0;                   /* Number (0 = don't grow, 1+ = grow proportionally) */
    
    /* Shrink factor */
    flex-shrink: 1;                 /* Number (0 = don't shrink, 1+ = shrink proportionally) */
    
    /* Initial size */
    flex-basis: auto;               /* auto, content, or any length value (px, %, etc.) */
    
    /* Shorthand for grow, shrink, basis */
    flex: 1;                        /* flex: 1 1 0% (grow shrink basis) */
    flex: 0 1 auto;                /* Default value */
    
    /* Individual alignment */
    align-self: auto;               /* auto, flex-start, flex-end, center, stretch, baseline */
    
    /* Order */
    order: 0;                       /* Number (lower numbers appear first) */
}
```

### Complete Example: Simple Flexbox Demo

**HTML (flexbox.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Flexbox Demo</title>
    <link rel="stylesheet" href="flexbox.css">
</head>
<body>
    <div class="container">
        <h1>Flexbox Basics</h1>
        
        <div class="demo-section">
            <h2>1. Basic Flex Container</h2>
            <div class="flex-container">
                <div class="box">Box 1</div>
                <div class="box">Box 2</div>
                <div class="box">Box 3</div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>2. Justify Content</h2>
            <div class="flex-container justify-center">
                <div class="box">Center</div>
                <div class="box">Center</div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>3. Align Items</h2>
            <div class="flex-container align-center">
                <div class="box">Item</div>
                <div class="box tall">Tall Item</div>
                <div class="box">Item</div>
            </div>
        </div>
        
        <div class="demo-section">
            <h2>4. Flex Direction</h2>
            <div class="flex-container direction-column">
                <div class="box">1</div>
                <div class="box">2</div>
                <div class="box">3</div>
            </div>
        </div>
    </div>
</body>
</html>
```

**CSS (flexbox.css):**
```css
/* Simple Flexbox Demo */
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
}

h1 {
    text-align: center;
    color: #333;
    margin-bottom: 30px;
}

.demo-section {
    margin-bottom: 40px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.demo-section h2 {
    color: #555;
    margin-bottom: 15px;
}

/* Basic flex container */
.flex-container {
    display: flex;
    background: #e3f2fd;
    padding: 20px;
    border-radius: 5px;
    min-height: 100px;
}

/* Basic box styling */
.box {
    width: 80px;
    height: 60px;
    background: #2196F3;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 5px;
    border-radius: 5px;
    font-weight: bold;
}

/* Justify content examples */
.justify-center {
    justify-content: center;
}

/* Align items examples */
.align-center {
    align-items: center;
}

.tall {
    height: 100px;
}

/* Flex direction examples */
.direction-column {
    flex-direction: column;
    height: 200px;
}
```

---

## 10. Responsive Design

- Text becomes too small to read on mobile
- Buttons are too tiny to tap
- Content overflows off the screen
- Users have to zoom and scroll horizontally

Responsive design ensures your site adapts to any screen size automatically.

### Media Queries
```css
/* Mobile first approach */
.responsive-container {
    width: 100%;
    padding: 10px;
}

/* Tablet styles */
@media (min-width: 768px) {
    .responsive-container {
        width: 80%;
        padding: 20px;
    }
}

/* Desktop styles */
@media (min-width: 1024px) {
    .responsive-container {
        width: 1200px;
        margin: 0 auto;
        padding: 30px;
    }
}
```

### Common Breakpoints
```css
/* Mobile */
@media (max-width: 767px) { /* styles */ }

/* Tablet */
@media (min-width: 768px) and (max-width: 1023px) { /* styles */ }

/* Desktop */
@media (min-width: 1024px) { /* styles */ }
```

### 📱 Complete Example: Responsive Gallery

**HTML (responsive.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Responsive Gallery</title>
    <link rel="stylesheet" href="responsive.css">
</head>
<body>
    <div class="responsive-showcase">
        <h1>📱 Responsive Design Gallery 📱</h1>
        <p class="intro">Resize your browser to see responsive design in action!</p>
        
        <nav class="responsive-nav">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">Gallery</a>
            <a href="#" class="nav-link">Contact</a>
        </nav>
        
        <div class="gallery-grid">
            <div class="gallery-item">
                <img src="https://via.placeholder.com/300x200/4CAF50/white?text=Image+1" alt="Gallery Image 1">
                <h3>Beautiful Landscape</h3>
                <p>Amazing view of mountains and lakes</p>
            </div>
            <div class="gallery-item">
                <img src="https://via.placeholder.com/300x200/2196F3/white?text=Image+2" alt="Gallery Image 2">
                <h3>City Skyline</h3>
                <p>Modern architecture at sunset</p>
            </div>
            <div class="gallery-item">
                <img src="https://via.placeholder.com/300x200/FF9800/white?text=Image+3" alt="Gallery Image 3">
                <h3>Ocean Waves</h3>
                <p>Peaceful beach scene</p>
            </div>
            <div class="gallery-item">
                <img src="https://via.placeholder.com/300x200/9C27B0/white?text=Image+4" alt="Gallery Image 4">
                <h3>Forest Path</h3>
                <p>Mysterious woodland trail</p>
            </div>
        </div>
        
        <div class="responsive-text">
            <h2>Responsive Design Features</h2>
            <div class="feature-grid">
                <div class="feature-item">
                    <h3>📱 Mobile First</h3>
                    <p>Designed for mobile devices first, then enhanced for larger screens</p>
                </div>
                <div class="feature-item">
                    <h3>🔄 Flexible Layout</h3>
                    <p>Uses flexible grids and images that adapt to any screen size</p>
                </div>
                <div class="feature-item">
                    <h3>⚡ Fast Loading</h3>
                    <p>Optimized for performance across all devices</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
                        <div class="flex-item">3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>space-between</h3>
                    <div class="flex-container justify-between">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>space-around</h3>
                    <div class="flex-container justify-around">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="align-items-demo">
            <h2>Align Items (Vertical Alignment)</h2>
            <div class="demo-grid">
                <div class="demo-item">
                    <h3>flex-start</h3>
                    <div class="flex-container align-start">
                        <div class="flex-item">Item 1</div>
                        <div class="flex-item tall">Tall Item</div>
                        <div class="flex-item">Item 3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>center</h3>
                    <div class="flex-container align-center">
                        <div class="flex-item">Item 1</div>
                        <div class="flex-item tall">Tall Item</div>
                        <div class="flex-item">Item 3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>flex-end</h3>
                    <div class="flex-container align-end">
                        <div class="flex-item">Item 1</div>
                        <div class="flex-item tall">Tall Item</div>
                        <div class="flex-item">Item 3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>stretch</h3>
                    <div class="flex-container align-stretch">
                        <div class="flex-item">Item 1</div>
                        <div class="flex-item">Item 2</div>
                        <div class="flex-item">Item 3</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="flex-direction-demo">
            <h2>Flex Direction</h2>
            <div class="demo-grid">
                <div class="demo-item">
                    <h3>row (default)</h3>
                    <div class="flex-container direction-row">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>column</h3>
                    <div class="flex-container direction-column">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>row-reverse</h3>
                    <div class="flex-container direction-row-reverse">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
                <div class="demo-item">
                    <h3>column-reverse</h3>
                    <div class="flex-container direction-column-reverse">
                        <div class="flex-item">1</div>
                        <div class="flex-item">2</div>
                        <div class="flex-item">3</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="flex-grow-demo">
            <h2>Flex Grow & Shrink</h2>
            <div class="flex-container grow-demo">
                <div class="flex-item grow-1">Grow: 1</div>
                <div class="flex-item grow-2">Grow: 2</div>
                <div class="flex-item grow-3">Grow: 3</div>
            </div>
        </section>
        
        <section class="interactive-flex">
            <h2>Interactive Flex Playground</h2>
            <div class="controls">
                <button onclick="changeJustify('flex-start')">Flex Start</button>
                <button onclick="changeJustify('center')">Center</button>
                <button onclick="changeJustify('space-between')">Space Between</button>
                <button onclick="changeJustify('space-around')">Space Around</button>
            </div>
            <div class="flex-container interactive" id="interactiveFlex">
                <div class="flex-item">🎨</div>
                <div class="flex-item">🎭</div>
                <div class="flex-item">🎪</div>
                <div class="flex-item">🎯</div>
            </div>
        </section>
    </div>
    
    <script>
        function changeJustify(value) {
            document.getElementById('interactiveFlex').style.justifyContent = value;
        }
    </script>
</body>
</html>
```

**CSS (flexbox.css):**
```css
/* Flexbox Fun House Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.flexbox-showcase {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

section {
    margin-bottom: 40px;
    padding: 25px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

/* Basic Flex Container */
.flex-container {
    display: flex;
    background: #e3f2fd;
    padding: 20px;
    border-radius: 10px;
    border: 2px dashed #2196f3;
    margin-bottom: 20px;
    min-height: 100px;
}

.flex-item {
    background: #2196f3;
    color: white;
    padding: 15px;
    margin: 5px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    transition: all 0.3s ease;
    min-width: 80px;
}

.flex-item:hover {
    background: #1976d2;
    transform: scale(1.05);
}

/* Demo Grid */
.demo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.demo-item {
    background: white;
    padding: 15px;
    border-radius: 10px;
    border: 2px solid #e9ecef;
}

.demo-item h3 {
    color: #495057;
    margin-bottom: 15px;
    text-align: center;
    font-size: 1.1em;
}

/* Justify Content */
.justify-start {
    justify-content: flex-start;
}

.justify-center {
    justify-content: center;
}

.justify-between {
    justify-content: space-between;
}

.justify-around {
    justify-content: space-around;
}

/* Align Items */
.align-start {
    align-items: flex-start;
}

.align-center {
    align-items: center;
}

.align-end {
    align-items: flex-end;
}

.align-stretch {
    align-items: stretch;
}

.tall {
    height: 80px;
    line-height: 50px;
}

/* Flex Direction */
.direction-row {
    flex-direction: row;
}

.direction-column {
    flex-direction: column;
    height: 200px;
}

.direction-row-reverse {
    flex-direction: row-reverse;
}

.direction-column-reverse {
    flex-direction: column-reverse;
    height: 200px;
}

/* Flex Grow Demo */
.grow-demo {
    height: 120px;
}

.grow-1 {
    flex-grow: 1;
    background: #4caf50;
}

.grow-2 {
    flex-grow: 2;
    background: #ff9800;
}

.grow-3 {
    flex-grow: 3;
    background: #f44336;
}

.grow-1:hover {
    background: #388e3c;
}

.grow-2:hover {
    background: #f57c00;
}

.grow-3:hover {
    background: #d32f2f;
}

/* Interactive Demo */
.interactive-flex {
    text-align: center;
}

.controls {
    margin-bottom: 20px;
}

.controls button {
    background: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    margin: 5px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    transition: background 0.3s ease;
}

.controls button:hover {
    background: #0056b3;
}

.interactive {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
    height: 150px;
    align-items: center;
}

.interactive .flex-item {
    background: rgba(255, 255, 255, 0.9);
    color: #2c3e50;
    font-size: 2em;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.interactive .flex-item:hover {
    background: white;
    transform: scale(1.1) rotate(5deg);
}
```

---

## 10. Responsive Design

**Why Responsive Design?** Your website needs to look great on phones, tablets, and desktops! Without responsive design:
- Text becomes too small to read on mobile
- Buttons are too tiny to tap
- Content overflows off the screen
- Users have to zoom and scroll horizontally

Responsive design ensures your site adapts to any screen size automatically.

### Media Queries
```css
/* Mobile first approach */
.responsive-container {
    width: 100%;
    padding: 10px;
}

/* Tablet styles */
@media (min-width: 768px) {
    .responsive-container {
        width: 80%;
        padding: 20px;
    }
}

/* Desktop styles */
@media (min-width: 1024px) {
    .responsive-container {
        width: 1200px;
        margin: 0 auto;
        padding: 30px;
    }
}
```

### Common Breakpoints
```css
/* Mobile */
@media (max-width: 767px) { /* styles */ }

/* Tablet */
@media (min-width: 768px) and (max-width: 1023px) { /* styles */ }

/* Desktop */
@media (min-width: 1024px) { /* styles */ }
```

### 📱 Complete Example: Responsive Gallery

**HTML (responsive.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Responsive Gallery</title>
    <link rel="stylesheet" href="responsive.css">
</head>
<body>
    <div class="responsive-showcase">
        <h1>📱 Responsive Design Gallery 📱</h1>
        <p class="intro">Resize your browser to see responsive design in action!</p>
        
        <nav class="responsive-nav">
            <a href="#" class="nav-item">Home</a>
            <a href="#" class="nav-item">About</a>
            <a href="#" class="nav-item">Gallery</a>
            <a href="#" class="nav-item">Contact</a>
        </nav>
        
        <section class="hero-section">
            <h2>Welcome to Our Gallery</h2>
            <p>This layout adapts to different screen sizes!</p>
        </section>
        
        <section class="gallery-grid">
            <div class="gallery-item">
                <div class="item-content">🎨 Art 1</div>
            </div>
            <div class="gallery-item">
                <div class="item-content">🎭 Art 2</div>
            </div>
            <div class="gallery-item">
                <div class="item-content">🎪 Art 3</div>
            </div>
            <div class="gallery-item">
                <div class="item-content">🎯 Art 4</div>
            </div>
            <div class="gallery-item">
                <div class="item-content">🌟 Art 5</div>
            </div>
            <div class="gallery-item">
                <div class="item-content">✨ Art 6</div>
            </div>
        </section>
        
        <section class="info-section">
            <div class="info-card">
                <h3>Mobile First</h3>
                <p>Design for mobile devices first, then enhance for larger screens.</p>
            </div>
            <div class="info-card">
                <h3>Flexible Layouts</h3>
                <p>Use flexible units like percentages and viewport units.</p>
            </div>
            <div class="info-card">
                <h3>Media Queries</h3>
                <p>Apply different styles based on screen size.</p>
            </div>
        </section>
    </div>
</body>
</html>
```

**CSS (responsive.css):**
```css
/* Responsive Gallery Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.responsive-showcase {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.1em;
    margin-bottom: 30px;
    font-style: italic;
}

/* Responsive Navigation */
.responsive-nav {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 30px;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
}

.nav-item {
    background: #3498db;
    color: white;
    padding: 12px 20px;
    text-decoration: none;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    transition: all 0.3s ease;
}

.nav-item:hover {
    background: #2980b9;
    transform: translateY(-2px);
}

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    color: white;
    border-radius: 15px;
    margin-bottom: 30px;
}

.hero-section h2 {
    font-size: 1.8em;
    margin-bottom: 15px;
}

/* Gallery Grid */
.gallery-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin-bottom: 30px;
}

.gallery-item {
    background: #f8f9fa;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.gallery-item:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

.item-content {
    padding: 40px 20px;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
    color: white;
    background: linear-gradient(45deg, #667eea, #764ba2);
}

/* Info Section */
.info-section {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
}

.info-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #3498db;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.info-card h3 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-size: 1.3em;
}

.info-card p {
    color: #6c757d;
    line-height: 1.6;
}

/* Tablet Styles */
@media (min-width: 768px) {
    .responsive-showcase {
        padding: 30px;
    }
    
    h1 {
        font-size: 2.5em;
    }
    
    .responsive-nav {
        flex-direction: row;
        justify-content: center;
    }
    
    .nav-item {
        flex: 1;
        max-width: 150px;
    }
    
    .gallery-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .info-section {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .hero-section h2 {
        font-size: 2.2em;
    }
}

/* Desktop Styles */
@media (min-width: 1024px) {
    .responsive-showcase {
        padding: 40px;
    }
    
    h1 {
        font-size: 3em;
    }
    
    .gallery-grid {
        grid-template-columns: repeat(3, 1fr);
    }
    
    .info-section {
        grid-template-columns: repeat(3, 1fr);
    }
    
    .hero-section {
        padding: 50px 40px;
    }
    
    .hero-section h2 {
        font-size: 2.5em;
    }
    
    .item-content {
        padding: 60px 20px;
        font-size: 1.8em;
    }
}

/* Large Desktop */
@media (min-width: 1200px) {
    .gallery-grid {
        grid-template-columns: repeat(4, 1fr);
    }
    
    .item-content {
        padding: 80px 20px;
        font-size: 2em;
    }
}
```

---

## 11. Common CSS Properties

**Why These Properties Matter?** These are the "building blocks" that make your website look professional:
- **Backgrounds**: Add colors, images, and gradients to make sections stand out
- **Borders**: Create visual separation between elements
- **Shadows**: Add depth and make elements "pop" off the page
- **Transforms**: Create cool effects like rotations and scaling
- **Transitions**: Make changes smooth and professional-looking

Without these, your site would look flat and boring!

### Backgrounds
```css
.background-examples {
    background-color: #f0f0f0;
    background-image: url('image.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    
    /* Shorthand */
    background: #f0f0f0 url('image.jpg') center/cover no-repeat;
}
```

### Borders
```css
.border-examples {
    border: 2px solid black;        /* width style color */
    border-radius: 10px;            /* Rounded corners */
    border-top: 1px dashed red;     /* Individual borders */
}
```

### Shadows
```css
.shadow-examples {
    /* Box shadow: x-offset y-offset blur-radius color */
    box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.3);
    
    /* Text shadow */
    text-shadow: 2px 2px 4px gray;
}
```

### 🎨 Complete Example: CSS Properties Showcase

**HTML (properties.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Properties Showcase</title>
    <link rel="stylesheet" href="properties.css">
</head>
<body>
    <div class="properties-showcase">
        <h1>🎨 CSS Properties Showcase 🎨</h1>
        <p class="intro">Explore common CSS properties and their effects!</p>
        
        <section class="background-demo">
            <h2>Background Properties</h2>
            <div class="demo-grid">
                <div class="bg-solid">Solid Color</div>
                <div class="bg-gradient">Gradient</div>
                <div class="bg-pattern">Pattern</div>
                <div class="bg-multiple">Multiple Backgrounds</div>
            </div>
        </section>
        
        <section class="border-demo">
            <h2>Border Properties</h2>
            <div class="demo-grid">
                <div class="border-solid">Solid Border</div>
                <div class="border-dashed">Dashed Border</div>
                <div class="border-dotted">Dotted Border</div>
                <div class="border-rounded">Rounded Border</div>
            </div>
        </section>
        
        <section class="shadow-demo">
            <h2>Shadow Properties</h2>
            <div class="demo-grid">
                <div class="shadow-box">Box Shadow</div>
                <div class="shadow-text">Text Shadow</div>
                <div class="shadow-inset">Inset Shadow</div>
                <div class="shadow-multiple">Multiple Shadows</div>
            </div>
        </section>
        
        <section class="transform-demo">
            <h2>Transform Properties</h2>
            <div class="demo-grid">
                <div class="transform-rotate">Rotate</div>
                <div class="transform-scale">Scale</div>
                <div class="transform-translate">Translate</div>
                <div class="transform-skew">Skew</div>
            </div>
        </section>
        
        <section class="transition-demo">
            <h2>Transition Properties</h2>
            <div class="demo-grid">
                <div class="transition-color">Color Transition</div>
                <div class="transition-transform">Transform Transition</div>
                <div class="transition-all">All Properties</div>
                <div class="transition-delay">Delayed Transition</div>
            </div>
        </section>
    </div>
</body>
</html>
```

**CSS (properties.css):**
```css
/* CSS Properties Showcase Styles */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
}

.properties-showcase {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 40px;
    font-style: italic;
}

section {
    margin-bottom: 40px;
    padding: 25px;
    background: #f8f9fa;
    border-radius: 15px;
    border-left: 5px solid #3498db;
}

section h2 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.8em;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

.demo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

/* Background Properties */
.bg-solid {
    background-color: #ff6b6b;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

.bg-gradient {
    background: linear-gradient(45deg, #4ecdc4, #45b7d1);
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

.bg-pattern {
    background: 
        radial-gradient(circle at 20% 50%, #ff6b6b 20%, transparent 20%),
        radial-gradient(circle at 80% 20%, #4ecdc4 20%, transparent 20%),
        radial-gradient(circle at 40% 80%, #45b7d1 20%, transparent 20%),
        #f8f9fa;
    background-size: 50px 50px;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
}

.bg-multiple {
    background: 
        linear-gradient(45deg, rgba(255, 107, 107, 0.3), rgba(78, 205, 196, 0.3)),
        url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="%23ff6b6b"/></svg>');
    background-size: cover, 20px 20px;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

/* Border Properties */
.border-solid {
    border: 3px solid #3498db;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    background: white;
}

.border-dashed {
    border: 3px dashed #e74c3c;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    background: white;
}

.border-dotted {
    border: 3px dotted #f39c12;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    background: white;
}

.border-rounded {
    border: 3px solid #27ae60;
    border-radius: 25px;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    background: white;
}

/* Shadow Properties */
.shadow-box {
    background: white;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

.shadow-text {
    background: white;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
}

.shadow-inset {
    background: #ecf0f1;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: inset 0 5px 10px rgba(0, 0, 0, 0.2);
}

.shadow-multiple {
    background: white;
    padding: 40px;
    text-align: center;
    color: #2c3e50;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 
        0 5px 10px rgba(255, 107, 107, 0.3),
        0 10px 20px rgba(78, 205, 196, 0.3),
        0 15px 30px rgba(69, 183, 209, 0.3);
}

/* Transform Properties */
.transform-rotate {
    background: #ff6b6b;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transform: rotate(15deg);
    transition: transform 0.3s ease;
}

.transform-rotate:hover {
    transform: rotate(0deg);
}

.transform-scale {
    background: #4ecdc4;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transform: scale(1.1);
    transition: transform 0.3s ease;
}

.transform-scale:hover {
    transform: scale(1.2);
}

.transform-translate {
    background: #45b7d1;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transform: translate(10px, 10px);
    transition: transform 0.3s ease;
}

.transform-translate:hover {
    transform: translate(0px, 0px);
}

.transform-skew {
    background: #96ceb4;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transform: skew(10deg, 5deg);
    transition: transform 0.3s ease;
}

.transform-skew:hover {
    transform: skew(0deg, 0deg);
}

/* Transition Properties */
.transition-color {
    background: #3498db;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: background-color 0.5s ease;
}

.transition-color:hover {
    background: #2980b9;
}

.transition-transform {
    background: #e74c3c;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: transform 0.5s ease;
}

.transition-transform:hover {
    transform: scale(1.1) rotate(5deg);
}

.transition-all {
    background: #f39c12;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: all 0.5s ease;
}

.transition-all:hover {
    background: #e67e22;
    transform: translateY(-10px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

.transition-delay {
    background: #9b59b6;
    padding: 40px;
    text-align: center;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: all 0.5s ease 0.2s;
}

.transition-delay:hover {
    background: #8e44ad;
    transform: scale(1.1);
}
```

---

## 12. Practical Example

**Why This Example?** This combines everything you've learned! It shows how all CSS concepts work together:
- **Typography**: Beautiful fonts and text styling
- **Colors**: Professional color scheme
- **Layout**: Flexbox for perfect alignment
- **Responsive**: Works on all devices
- **Interactions**: Hover effects and smooth transitions

This is what a real website looks like when you apply all CSS knowledge!

### HTML Structure
```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header class="header">
        <h1>My Website</h1>
        <nav class="navigation">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">Contact</a>
        </nav>
    </header>
    
    <main class="main-content">
        <section class="hero">
            <h2>Welcome to My Site</h2>
            <p class="hero-text">This is a sample website built with HTML and CSS.</p>
            <button class="cta-button">Get Started</button>
        </section>
    </main>
</body>
</html>
```

### CSS Styles
```css
/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
}

/* Header styles */
.header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navigation {
    display: flex;
    gap: 2rem;
}

.nav-link {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

.nav-link:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

/* Main content */
.main-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.hero {
    text-align: center;
    padding: 4rem 0;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.hero-text {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    color: #666;
}

.cta-button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.cta-button:hover {
    background-color: #2980b9;
}

/* Responsive design */
@media (max-width: 768px) {
    .header {
        flex-direction: column;
        gap: 1rem;
    }
    
    .hero h2 {
        font-size: 2rem;
    }
    
    .main-content {
        padding: 1rem;
    }
}
```

### 🚀 Complete Example: Modern Portfolio Website

**HTML (portfolio.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Portfolio</title>
    <link rel="stylesheet" href="portfolio.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo">Portfolio</h1>
            <nav class="navigation">
                <a href="#home" class="nav-link">Home</a>
                <a href="#about" class="nav-link">About</a>
                <a href="#projects" class="nav-link">Projects</a>
                <a href="#contact" class="nav-link">Contact</a>
            </nav>
        </div>
    </header>
    
    <main class="main-content">
        <section id="home" class="hero">
            <div class="container">
                <h2 class="hero-title">Welcome to My Portfolio</h2>
                <p class="hero-subtitle">I create amazing web experiences</p>
                <button class="cta-button">View My Work</button>
            </div>
        </section>
        
        <section id="about" class="about">
            <div class="container">
                <h2 class="section-title">About Me</h2>
                <div class="about-content">
                    <div class="about-text">
                        <p>I'm a passionate web developer with expertise in modern CSS and responsive design.</p>
                        <p>I love creating beautiful, functional websites that provide great user experiences.</p>
                    </div>
                    <div class="skills">
                        <div class="skill-item">HTML5</div>
                        <div class="skill-item">CSS3</div>
                        <div class="skill-item">JavaScript</div>
                        <div class="skill-item">React</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section id="projects" class="projects">
            <div class="container">
                <h2 class="section-title">My Projects</h2>
                <div class="project-grid">
                    <div class="project-card">
                        <div class="project-image">🎨</div>
                        <h3>Art Gallery</h3>
                        <p>A responsive gallery showcasing digital artwork.</p>
                    </div>
                    <div class="project-card">
                        <div class="project-image">🛒</div>
                        <h3>E-commerce Site</h3>
                        <p>Modern online store with smooth animations.</p>
                    </div>
                    <div class="project-card">
                        <div class="project-image">📱</div>
                        <h3>Mobile App</h3>
                        <p>Cross-platform mobile application.</p>
                    </div>
                </div>
            </div>
        </section>
        
        <section id="contact" class="contact">
            <div class="container">
                <h2 class="section-title">Get In Touch</h2>
                <div class="contact-content">
                    <div class="contact-info">
                        <h3>Let's work together!</h3>
                        <p>I'm always interested in new projects and opportunities.</p>
                    </div>
                    <form class="contact-form">
                        <input type="text" placeholder="Your Name" required>
                        <input type="email" placeholder="Your Email" required>
                        <textarea placeholder="Your Message" rows="5" required></textarea>
                        <button type="submit" class="submit-btn">Send Message</button>
                    </form>
                </div>
            </div>
        </section>
    </main>
    
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 Portfolio. All rights reserved.</p>
        </div>
    </footer>
</body>
</html>
```

**CSS (portfolio.css):**
```css
/* Modern Portfolio Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    overflow-x: hidden;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header Styles */
.header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 20px;
}

.logo {
    font-size: 1.8rem;
    font-weight: bold;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.navigation {
    display: flex;
    gap: 2rem;
}

.nav-link {
    color: #333;
    text-decoration: none;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    transition: all 0.3s ease;
    position: relative;
}

.nav-link:hover {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    transform: translateY(-2px);
}

/* Main Content */
.main-content {
    margin-top: 80px;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 6rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="rgba(255,255,255,0.1)"/></svg>');
    background-size: 50px 50px;
    animation: float 20s infinite linear;
}

@keyframes float {
    0% { transform: translateY(0px); }
    100% { transform: translateY(-50px); }
}

.hero-title {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.hero-subtitle {
    font-size: 1.3rem;
    margin-bottom: 2rem;
    opacity: 0.9;
}

.cta-button {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 2px solid white;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.cta-button:hover {
    background: white;
    color: #667eea;
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

/* About Section */
.about {
    padding: 5rem 0;
    background: #f8f9fa;
}

.section-title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 3rem;
    color: #2c3e50;
    position: relative;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 50px;
    height: 3px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    border-radius: 2px;
}

.about-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    align-items: center;
}

.about-text p {
    font-size: 1.1rem;
    margin-bottom: 1rem;
    color: #6c757d;
}

.skills {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

.skill-item {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    color: #667eea;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.skill-item:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

/* Projects Section */
.projects {
    padding: 5rem 0;
    background: white;
}

.project-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.project-card {
    background: white;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.project-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
}

.project-image {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    padding: 3rem;
    text-align: center;
    font-size: 3rem;
    color: white;
}

.project-card h3 {
    padding: 1.5rem 1.5rem 0.5rem;
    color: #2c3e50;
    font-size: 1.3rem;
}

.project-card p {
    padding: 0 1.5rem 1.5rem;
    color: #6c757d;
}

/* Contact Section */
.contact {
    padding: 5rem 0;
    background: #f8f9fa;
}

.contact-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    align-items: start;
}

.contact-info h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.contact-info p {
    color: #6c757d;
    font-size: 1.1rem;
}

.contact-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.contact-form input,
.contact-form textarea {
    padding: 1rem;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.contact-form input:focus,
.contact-form textarea:focus {
    outline: none;
    border-color: #667eea;
}

.submit-btn {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 10px;
    font-size: 1.1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.submit-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
}

/* Footer */
.footer {
    background: #2c3e50;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .header .container {
        flex-direction: column;
        gap: 1rem;
    }
    
    .navigation {
        gap: 1rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
    }
    
    .about-content,
    .contact-content {
        grid-template-columns: 1fr;
        gap: 2rem;
    }
    
    .skills {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .project-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .section-title {
        font-size: 2rem;
    }
    
    .skills {
        grid-template-columns: 1fr;
    }
}
```

---

## 13. Best Practices

**Why Follow Best Practices?** Writing CSS is easy, but writing GOOD CSS is an art! Best practices help you:
- **Write maintainable code**: Others (and future you) can understand and modify it
- **Avoid common mistakes**: Prevent layout bugs and performance issues
- **Work efficiently**: Use tools and techniques that save time
- **Create professional websites**: Code that looks and works like industry standards

Following these practices separates beginners from professional developers!

### Organization
- Use external stylesheets for better maintainability
- Group related styles together
- Use comments to organize your code
- Follow a consistent naming convention

### Performance
- Minimize CSS file size
- Use shorthand properties when possible
- Avoid overly specific selectors
- Combine similar selectors

### Code Quality
```css
/* Good - Clear and maintainable */
.navigation-menu {
    display: flex;
    gap: 1rem;
}

.navigation-menu a {
    padding: 0.5rem 1rem;
    text-decoration: none;
}

/* Avoid - Too specific and hard to maintain */
div#header ul.navigation li a.menu-item {
    padding: 8px 16px;
}
```

### 🎯 Complete Example: CSS Best Practices Demo

**HTML (best-practices.html):**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Best Practices Demo</title>
    <link rel="stylesheet" href="best-practices.css">
</head>
<body>
    <div class="best-practices-showcase">
        <h1>🎯 CSS Best Practices Demo 🎯</h1>
        <p class="intro">Learn how to write clean, maintainable CSS code!</p>
        
        <section class="organization-demo">
            <h2>1. Code Organization</h2>
            <div class="demo-content">
                <div class="good-example">
                    <h3>✅ Good Organization</h3>
                    <div class="well-organized">
                        <nav class="main-nav">
                            <a href="#" class="nav-link">Home</a>
                            <a href="#" class="nav-link">About</a>
                            <a href="#" class="nav-link">Contact</a>
                        </nav>
                    </div>
                </div>
                <div class="bad-example">
                    <h3>❌ Poor Organization</h3>
                    <div class="poorly-organized">
                        <nav class="nav">
                            <a href="#" class="link">Home</a>
                            <a href="#" class="link">About</a>
                            <a href="#" class="link">Contact</a>
                        </nav>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="naming-demo">
            <h2>2. Naming Conventions</h2>
            <div class="demo-content">
                <div class="good-example">
                    <h3>✅ BEM Methodology</h3>
                    <div class="bem-example">
                        <div class="card">
                            <div class="card__header">
                                <h3 class="card__title">Card Title</h3>
                            </div>
                            <div class="card__body">
                                <p class="card__text">Card content goes here.</p>
                            </div>
                            <div class="card__footer">
                                <button class="card__button card__button--primary">Action</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="bad-example">
                    <h3>❌ Inconsistent Naming</h3>
                    <div class="inconsistent-example">
                        <div class="cardBox">
                            <div class="cardHeader">
                                <h3 class="cardTitle">Card Title</h3>
                            </div>
                            <div class="cardBody">
                                <p class="cardText">Card content goes here.</p>
                            </div>
                            <div class="cardFooter">
                                <button class="btn btn-primary">Action</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="performance-demo">
            <h2>3. Performance Optimization</h2>
            <div class="demo-content">
                <div class="good-example">
                    <h3>✅ Optimized CSS</h3>
                    <div class="optimized">
                        <div class="box">Box 1</div>
                        <div class="box">Box 2</div>
                        <div class="box">Box 3</div>
                    </div>
                </div>
                <div class="bad-example">
                    <h3>❌ Unoptimized CSS</h3>
                    <div class="unoptimized">
                        <div class="box1">Box 1</div>
                        <div class="box2">Box 2</div>
                        <div class="box3">Box 3</div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="maintainability-demo">
            <h2>4. Maintainability</h2>
            <div class="demo-content">
                <div class="good-example">
                    <h3>✅ Maintainable Code</h3>
                    <div class="maintainable">
                        <button class="btn btn--primary">Primary Button</button>
                        <button class="btn btn--secondary">Secondary Button</button>
                        <button class="btn btn--success">Success Button</button>
                    </div>
                </div>
                <div class="bad-example">
                    <h3>❌ Hard to Maintain</h3>
                    <div class="hard-to-maintain">
                        <button class="primaryButton">Primary Button</button>
                        <button class="secondaryButton">Secondary Button</button>
                        <button class="successButton">Success Button</button>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="tips-section">
            <h2>💡 Pro Tips</h2>
            <div class="tips-grid">
                <div class="tip-card">
                    <h3>Use CSS Variables</h3>
                    <p>Define reusable values for colors, fonts, and spacing.</p>
                </div>
                <div class="tip-card">
                    <h3>Mobile First</h3>
                    <p>Start with mobile styles and enhance for larger screens.</p>
                </div>
                <div class="tip-card">
                    <h3>Use Flexbox/Grid</h3>
                    <p>Modern layout methods are more powerful and flexible.</p>
                </div>
                <div class="tip-card">
                    <h3>Keep It Simple</h3>
                    <p>Avoid over-engineering. Simple solutions are often best.</p>
                </div>
            </div>
        </section>
    </div>
</body>
</html>
```

**CSS (best-practices.css):**
```css
/* CSS Best Practices Demo Styles */

/* CSS Variables for consistency */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #4ecdc4;
    --text-color: #2c3e50;
    --text-light: #6c757d;
    --bg-light: #f8f9fa;
    --border-radius: 10px;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 2rem;
    --spacing-xl: 3rem;
    --font-size-sm: 0.9rem;
    --font-size-md: 1rem;
    --font-size-lg: 1.2rem;
    --font-size-xl: 2rem;
}

/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: var(--spacing-md);
}

.best-practices-showcase {
    max-width: 1200px;
    margin: 0 auto;
    background: rgba(255, 255, 255, 0.95);
    padding: var(--spacing-xl);
    border-radius: var(--border-radius);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    color: var(--text-color);
    font-size: var(--font-size-xl);
    margin-bottom: var(--spacing-sm);
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.intro {
    text-align: center;
    color: var(--text-light);
    font-size: var(--font-size-lg);
    margin-bottom: var(--spacing-xl);
    font-style: italic;
}

section {
    margin-bottom: var(--spacing-xl);
    padding: var(--spacing-lg);
    background: var(--bg-light);
    border-radius: var(--border-radius);
    border-left: 5px solid var(--primary-color);
}

section h2 {
    color: var(--text-color);
    margin-bottom: var(--spacing-lg);
    font-size: 1.8rem;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: var(--spacing-sm);
}

.demo-content {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--spacing-lg);
}

.good-example,
.bad-example {
    background: white;
    padding: var(--spacing-lg);
    border-radius: var(--border-radius);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.good-example h3 {
    color: var(--success-color);
    margin-bottom: var(--spacing-md);
}

.bad-example h3 {
    color: #e74c3c;
    margin-bottom: var(--spacing-md);
}

/* Good Organization Example */
.main-nav {
    display: flex;
    gap: var(--spacing-md);
    background: var(--primary-color);
    padding: var(--spacing-md);
    border-radius: var(--border-radius);
}

.nav-link {
    color: white;
    text-decoration: none;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: calc(var(--border-radius) / 2);
    transition: background-color 0.3s ease;
}

.nav-link:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

/* Poor Organization Example */
.nav {
    display: flex;
    gap: 1rem;
    background: #e74c3c;
    padding: 0.5rem;
    border-radius: 5px;
}

.link {
    color: white;
    text-decoration: none;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
}

/* BEM Methodology Example */
.card {
    background: white;
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.card__header {
    background: var(--primary-color);
    padding: var(--spacing-md);
    color: white;
}

.card__title {
    font-size: var(--font-size-lg);
    margin: 0;
}

.card__body {
    padding: var(--spacing-md);
}

.card__text {
    color: var(--text-light);
    margin: 0;
}

.card__footer {
    padding: var(--spacing-md);
    background: var(--bg-light);
    text-align: center;
}

.card__button {
    background: var(--primary-color);
    color: white;
    border: none;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: calc(var(--border-radius) / 2);
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.card__button--primary {
    background: var(--primary-color);
}

.card__button:hover {
    opacity: 0.9;
}

/* Inconsistent Naming Example */
.cardBox {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.cardHeader {
    background: #e74c3c;
    padding: 1rem;
    color: white;
}

.cardTitle {
    font-size: 1.2rem;
    margin: 0;
}

.cardBody {
    padding: 1rem;
}

.cardText {
    color: #6c757d;
    margin: 0;
}

.cardFooter {
    padding: 1rem;
    background: #f8f9fa;
    text-align: center;
}

.btn {
    background: #e74c3c;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
}

.btn-primary {
    background: #e74c3c;
}

/* Optimized CSS Example */
.optimized {
    display: flex;
    gap: var(--spacing-md);
}

.box {
    background: var(--success-color);
    color: white;
    padding: var(--spacing-md);
    border-radius: var(--border-radius);
    text-align: center;
    flex: 1;
}

/* Unoptimized CSS Example */
.unoptimized {
    display: flex;
    gap: 1rem;
}

.box1,
.box2,
.box3 {
    background: #e74c3c;
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    flex: 1;
}

/* Maintainable Code Example */
.maintainable {
    display: flex;
    gap: var(--spacing-md);
    flex-wrap: wrap;
}

.btn {
    padding: var(--spacing-sm) var(--spacing-md);
    border: none;
    border-radius: calc(var(--border-radius) / 2);
    cursor: pointer;
    font-size: var(--font-size-md);
    transition: all 0.3s ease;
}

.btn--primary {
    background: var(--primary-color);
    color: white;
}

.btn--secondary {
    background: var(--text-light);
    color: white;
}

.btn--success {
    background: var(--success-color);
    color: white;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

/* Hard to Maintain Example */
.hard-to-maintain {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.primaryButton {
    background: #e74c3c;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.secondaryButton {
    background: #6c757d;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.successButton {
    background: #4ecdc4;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

/* Tips Section */
.tips-section {
    background: linear-gradient(45deg, var(--success-color), var(--primary-color));
    color: white;
}

.tips-section h2 {
    color: white;
    border-bottom-color: white;
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-md);
}

.tip-card {
    background: rgba(255, 255, 255, 0.1);
    padding: var(--spacing-md);
    border-radius: var(--border-radius);
    backdrop-filter: blur(10px);
}

.tip-card h3 {
    margin-bottom: var(--spacing-sm);
    font-size: var(--font-size-lg);
}

.tip-card p {
    opacity: 0.9;
    line-height: 1.5;
}

/* Responsive Design */
@media (max-width: 768px) {
    .demo-content {
        grid-template-columns: 1fr;
    }
    
    .tips-grid {
        grid-template-columns: 1fr;
    }
    
    h1 {
        font-size: 1.5rem;
    }
    
    section {
        padding: var(--spacing-md);
    }
}
```

---

## 14. Next Steps

### Continue Learning
- CSS Grid for advanced layouts
- CSS animations and transitions
- CSS preprocessors (Sass, Less)
- CSS frameworks (Bootstrap, Tailwind)
- Modern CSS features (CSS Variables, Flexbox Grid)

### Practice Projects
- Build a personal portfolio website
- Create a responsive navigation menu
- Design a product card component
- Make a simple landing page

### Resources
- MDN Web Docs for CSS reference
- CSS-Tricks for tutorials and tips
- CodePen for experimenting with code
- Can I Use for browser compatibility

---
