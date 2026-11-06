# HTML Lectures for Web Development

## Lecture 1: Introduction to HTML

### What is HTML?

HTML (HyperText Markup Language) is the standard markup language used to create web pages. Think of it as the skeleton or structure of a webpage - it defines what content appears where and how it's organized.

**Key Points:**
- HTML is not a programming language, it's a markup language
- It uses tags to define different types of content
- HTML tells the browser how to display content
- Every website you visit uses HTML as its foundation

### Why Learn HTML?

1. **Foundation of Web Development** - Everything on the web starts with HTML
2. **Universal Language** - All browsers understand HTML
3. **Simple to Learn** - Uses plain English words as commands
4. **Essential Skill** - Required for any web development career

### How HTML Works

HTML works by using **tags** to wrap around content and tell the browser what that content is. Tags are like labels that describe the content inside them.

**Basic Structure:**
```html
<tagname>Content goes here</tagname>
```

### Your First HTML Document

```html
<!DOCTYPE html>
<html>
<head>
    <title>My First Web Page</title>
</head>
<body>
    <h1>Welcome to My Website!</h1>
    <p>This is my first paragraph.</p>
</body>
</html>
```

**Explanation:**
- `<!DOCTYPE html>` - Tells the browser this is HTML5
- `<html>` - Root element that contains all content
- `<head>` - Contains information about the page (not visible)
- `<title>` - Sets the page title (appears in browser tab)
- `<body>` - Contains all visible content
- `<h1>` - Main heading
- `<p>` - Paragraph text

---

## Lecture 2: HTML Document Structure

### The HTML5 Document Structure

Every HTML document follows a specific structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
</head>
<body>
    <!-- Your content goes here -->
</body>
</html>
```

### Understanding Each Part

#### 1. DOCTYPE Declaration
```html
<!DOCTYPE html>
```
- Must be the first line
- Tells browser which version of HTML to use
- HTML5 uses simple `<!DOCTYPE html>`

#### 2. HTML Element
```html
<html lang="en">
```
- Root element of the page
- `lang="en"` specifies the language (English)
- All other elements go inside this

#### 3. Head Section
```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
</head>
```
- Contains metadata (information about the page)
- Not visible to users
- Includes title, character encoding, styles, scripts

#### 4. Body Section
```html
<body>
    <!-- All visible content goes here -->
</body>
```
- Contains all visible content
- Everything users see on the webpage

### Comments in HTML

Comments help you document your code:
```html
<!-- This is a comment -->
<!-- Comments are not displayed in the browser -->
```

---

## Lecture 3: HTML Tags and Elements

### What are Tags and Elements?

**Tag:** The markup itself (e.g., `<h1>`, `</h1>`)
**Element:** The complete structure including tags and content

```html
<h1>This is a heading</h1>
```
- `<h1>` - Opening tag
- `</h1>` - Closing tag
- The whole thing is an "h1 element"

### Types of Tags

#### 1. Container Tags (Most Common)
Have opening and closing tags:
```html
<h1>Heading</h1>
<p>Paragraph</p>
<div>Division</div>
```

#### 2. Self-Closing Tags
Don't have closing tags:
```html
<br>     <!-- Line break -->
<hr>     <!-- Horizontal rule -->
<img>    <!-- Image -->
<input>  <!-- Input field -->
```

### Basic Text Tags

#### Headings
```html
<h1>Main Heading (Largest)</h1>
<h2>Sub Heading</h2>
<h3>Section Heading</h3>
<h4>Subsection Heading</h4>
<h5>Minor Heading</h5>
<h6>Smallest Heading</h6>
```

#### Paragraphs and Text Formatting
```html
<p>This is a paragraph.</p>
<p>This is another paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>

<b>Bold text (visual only)</b>
<i>Italic text (visual only)</i>
<u>Underlined text</u>
<small>Small text</small>
<mark>Highlighted text</mark>
```

#### Line Breaks and Horizontal Rules
```html
<p>First line<br>Second line</p>
<hr>
<p>Text after horizontal line</p>
```

### Practice Example
```html
<!DOCTYPE html>
<html>
<head>
    <title>Text Formatting Demo</title>
</head>
<body>
    <h1>Welcome to My Blog</h1>
    <h2>About Web Development</h2>
    
    <p>Web development is <strong>exciting</strong> and <em>creative</em>!</p>
    <p>HTML is the <mark>foundation</mark> of every website.</p>
    
    <hr>
    
    <h3>Why Learn HTML?</h3>
    <p>HTML is <small>(HyperText Markup Language)</small> essential for:</p>
</body>
</html>
```

---

## Lecture 4: Working with Lists

Lists are essential for organizing information on web pages. HTML provides three types of lists.

### 1. Unordered Lists (Bullet Points)

```html
<ul>
    <li>First item</li>
    <li>Second item</li>
    <li>Third item</li>
</ul>
```

**Result:**
• First item
• Second item  
• Third item

### 2. Ordered Lists (Numbered)

```html
<ol>
    <li>Step one</li>
    <li>Step two</li>
    <li>Step three</li>
</ol>
```

**Result:**
1. Step one
2. Step two
3. Step three

### 3. Description Lists (Term and Definition)

```html
<dl>
    <dt>HTML</dt>
    <dd>HyperText Markup Language</dd>
    
    <dt>CSS</dt>
    <dd>Cascading Style Sheets</dd>
    
    <dt>JavaScript</dt>
    <dd>Programming language for web interactivity</dd>
</dl>
```

### Nested Lists

You can put lists inside lists:

```html
<ul>
    <li>Web Development
        <ul>
            <li>Frontend
                <ol>
                    <li>HTML</li>
                    <li>CSS</li>
                    <li>JavaScript</li>
                </ol>
            </li>
            <li>Backend
                <ul>
                    <li>Python</li>
                    <li>Node.js</li>
                    <li>PHP</li>
                </ul>
            </li>
        </ul>
    </li>
    <li>Mobile Development</li>
    <li>Desktop Development</li>
</ul>
```

### Practical Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>HTML Tutorial Topics</title>
</head>
<body>
    <h1>Web Development Learning Path</h1>
    
    <h2>Frontend Technologies</h2>
    <ol>
        <li>HTML - Structure</li>
        <li>CSS - Styling</li>
        <li>JavaScript - Behavior</li>
    </ol>
    
    <h2>What You'll Learn</h2>
    <ul>
        <li>Creating web pages</li>
        <li>Styling with CSS</li>
        <li>Adding interactivity</li>
        <li>Responsive design</li>
    </ul>
    
    <h2>HTML Glossary</h2>
    <dl>
        <dt>Element</dt>
        <dd>A complete HTML structure with opening and closing tags</dd>
        
        <dt>Attribute</dt>
        <dd>Additional information added to HTML tags</dd>
        
        <dt>Tag</dt>
        <dd>The markup code like &lt;h1&gt; or &lt;p&gt;</dd>
    </dl>
</body>
</html>
```

---

## Lecture 5: Links and Navigation

Links are what make the web "web" - they connect pages together and allow navigation.

### Basic Links

#### External Links (to other websites)
```html
<a href="https://www.google.com">Visit Google</a>
<a href="https://www.github.com" target="_blank">Open GitHub in New Tab</a>
```

#### Internal Links (to other pages on your site)
```html
<a href="about.html">About Us</a>
<a href="contact.html">Contact</a>
<a href="../index.html">Back to Home</a>
```

#### Links within the same page
```html
<a href="#section1">Go to Section 1</a>
<a href="#top">Back to Top</a>

<h2 id="section1">Section 1</h2>
<p>Content here...</p>
```

### Email and Phone Links

```html
<a href="mailto:someone@example.com">Send Email</a>
<a href="tel:+1234567890">Call Us</a>
```

### Link Attributes

#### target Attribute
```html
<a href="https://example.com" target="_blank">New Window</a>
<a href="https://example.com" target="_self">Same Window (default)</a>
```

#### title Attribute (shows tooltip)
```html
<a href="https://example.com" title="Visit Example Website">Example</a>
```

### Navigation Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Navigation Example</title>
</head>
<body>
    <!-- Navigation Menu -->
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#services">Services</a></li>
            <li><a href="#contact">Contact</a></li>
        </ul>
    </nav>
    
    <!-- Page Sections -->
    <section id="home">
        <h1>Welcome to Our Website</h1>
        <p>This is the home section.</p>
    </section>
    
    <section id="about">
        <h2>About Us</h2>
        <p>Learn more about our company.</p>
        <a href="https://www.example.com" target="_blank">Visit our main site</a>
    </section>
    
    <section id="services">
        <h2>Our Services</h2>
        <p>What we offer to our clients.</p>
    </section>
    
    <section id="contact">
        <h2>Contact Information</h2>
        <p>Email: <a href="mailto:info@example.com">info@example.com</a></p>
        <p>Phone: <a href="tel:+1234567890">+1 (234) 567-890</a></p>
    </section>
    
    <p><a href="#home">Back to Top</a></p>
</body>
</html>
```

---

## Lecture 6: Images in HTML

Images make websites visually appealing and help communicate information effectively.

### Basic Image Syntax

```html
<img src="image.jpg" alt="Description of image">
```

**Required Attributes:**
- `src` - Path to the image file
- `alt` - Alternative text for accessibility

### Image Sources

#### Local Images (on your server)
```html
<img src="images/logo.png" alt="Company Logo">
<img src="../photos/team.jpg" alt="Our Team">
```

#### External Images (from other websites)
```html
<img src="https://example.com/image.jpg" alt="Example Image">
```

### Image Attributes

#### Size Control
```html
<img src="photo.jpg" alt="Photo" width="300" height="200">
<img src="photo.jpg" alt="Photo" width="50%">
```

#### Title (tooltip on hover)
```html
<img src="photo.jpg" alt="Photo" title="This appears on hover">
```

### Image Best Practices

1. **Always use alt text** for accessibility
2. **Optimize file sizes** for faster loading
3. **Use appropriate formats:**
   - JPG for photos
   - PNG for graphics with transparency
   - SVG for simple graphics and icons

### Figures and Captions

```html
<figure>
    <img src="chart.png" alt="Sales data chart">
    <figcaption>Sales increased by 25% this quarter</figcaption>
</figure>
```

### Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Image Gallery</title>
</head>
<body>
    <h1>Our Photo Gallery</h1>
    
    <!-- Company Logo -->
    <img src="images/logo.png" alt="TechCorp Logo" width="200">
    
    <h2>Featured Photos</h2>
    
    <!-- Photo with caption -->
    <figure>
        <img src="images/office.jpg" alt="Modern office space with computers" width="400">
        <figcaption>Our modern office in downtown</figcaption>
    </figure>
    
    <!-- Team photos -->
    <h3>Meet Our Team</h3>
    <img src="images/team1.jpg" alt="Development team working together" width="300" title="Our amazing developers">
    <img src="images/team2.jpg" alt="Marketing team in meeting" width="300" title="Creative marketing team">
    
    <!-- External image example -->
    <h3>Inspiration</h3>
    <img src="https://via.placeholder.com/300x200" alt="Placeholder inspiration image" width="300">
    
    <p>All images are optimized for web viewing.</p>
</body>
</html>
```

---

## Lecture 7: Tables in HTML

Tables are used to display data in rows and columns, like spreadsheets.

### Basic Table Structure

```html
<table>
    <tr>
        <td>Cell 1</td>
        <td>Cell 2</td>
    </tr>
    <tr>
        <td>Cell 3</td>
        <td>Cell 4</td>
    </tr>
</table>
```

**Key Elements:**
- `<table>` - Container for the entire table
- `<tr>` - Table row
- `<td>` - Table data cell

### Table with Headers

```html
<table>
    <tr>
        <th>Name</th>
        <th>Age</th>
        <th>City</th>
    </tr>
    <tr>
        <td>John</td>
        <td>25</td>
        <td>New York</td>
    </tr>
    <tr>
        <td>Sarah</td>
        <td>30</td>
        <td>London</td>
    </tr>
</table>
```

### Proper Table Structure

```html
<table>
    <thead>
        <tr>
            <th>Product</th>
            <th>Price</th>
            <th>Stock</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Laptop</td>
            <td>$999</td>
            <td>15</td>
        </tr>
        <tr>
            <td>Mouse</td>
            <td>$25</td>
            <td>50</td>
        </tr>
    </tbody>
</table>
```

### Advanced Table Features

#### Spanning Cells
```html
<table>
    <tr>
        <th colspan="2">Sales Report</th>
    </tr>
    <tr>
        <td>Q1</td>
        <td>$10,000</td>
    </tr>
    <tr>
        <td>Q2</td>
        <td>$15,000</td>
    </tr>
</table>
```

#### Row Spanning
```html
<table>
    <tr>
        <td rowspan="2">Products</td>
        <td>Laptop</td>
        <td>$999</td>
    </tr>
    <tr>
        <td>Phone</td>
        <td>$599</td>
    </tr>
</table>
```

### Complete Table Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Student Grades</title>
</head>
<body>
    <h1>Class Grade Report</h1>
    
    <table border="1">
        <thead>
            <tr>
                <th>Student ID</th>
                <th>Name</th>
                <th>Math</th>
                <th>Science</th>
                <th>English</th>
                <th>Average</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>001</td>
                <td>Alice Johnson</td>
                <td>85</td>
                <td>92</td>
                <td>88</td>
                <td>88.3</td>
            </tr>
            <tr>
                <td>002</td>
                <td>Bob Smith</td>
                <td>78</td>
                <td>85</td>
                <td>82</td>
                <td>81.7</td>
            </tr>
            <tr>
                <td>003</td>
                <td>Carol Davis</td>
                <td>95</td>
                <td>89</td>
                <td>94</td>
                <td>92.7</td>
            </tr>
        </tbody>
        <tfoot>
            <tr>
                <td colspan="5"><strong>Class Average</strong></td>
                <td><strong>87.6</strong></td>
            </tr>
        </tfoot>
    </table>
    
    <p><em>Note: Grades are out of 100 points</em></p>
</body>
</html>
```

---

## Lecture 8: HTML Forms

Forms allow users to input and submit data to websites.

### Basic Form Structure

```html
<form action="/submit" method="POST">
    <!-- Form elements go here -->
    <input type="submit" value="Submit">
</form>
```

**Key Attributes:**
- `action` - Where to send the data
- `method` - How to send it (GET or POST)

### Input Types

#### Text Inputs
```html
<label for="name">Name:</label>
<input type="text" id="name" name="name">

<label for="email">Email:</label>
<input type="email" id="email" name="email" required>

<label for="password">Password:</label>
<input type="password" id="password" name="password">
```

#### Other Input Types
```html
<input type="number" name="age" min="1" max="100">
<input type="date" name="birthday">
<input type="tel" name="phone">
<input type="url" name="website">
<input type="color" name="favorite-color">
<input type="range" name="volume" min="0" max="100">
```

### Selection Elements

#### Radio Buttons (choose one)
```html
<fieldset>
    <legend>Choose your size:</legend>
    <input type="radio" id="small" name="size" value="small">
    <label for="small">Small</label>
    
    <input type="radio" id="medium" name="size" value="medium" checked>
    <label for="medium">Medium</label>
    
    <input type="radio" id="large" name="size" value="large">
    <label for="large">Large</label>
</fieldset>
```

#### Checkboxes (choose multiple)
```html
<fieldset>
    <legend>Select your interests:</legend>
    <input type="checkbox" id="sports" name="interests" value="sports">
    <label for="sports">Sports</label>
    
    <input type="checkbox" id="music" name="interests" value="music">
    <label for="music">Music</label>
    
    <input type="checkbox" id="movies" name="interests" value="movies">
    <label for="movies">Movies</label>
</fieldset>
```

#### Dropdown Select
```html
<label for="country">Country:</label>
<select id="country" name="country">
    <option value="">Choose a country</option>
    <option value="us">United States</option>
    <option value="uk">United Kingdom</option>
    <option value="ca">Canada</option>
    <option value="au">Australia</option>
</select>
```

### Text Areas

```html
<label for="message">Message:</label>
<textarea id="message" name="message" rows="4" cols="50" placeholder="Enter your message here..."></textarea>
```

### Complete Form Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Contact Form</title>
</head>
<body>
    <h1>Contact Us</h1>
    
    <form action="/contact" method="POST">
        <!-- Personal Information -->
        <fieldset>
            <legend>Personal Information</legend>
            
            <label for="firstName">First Name:</label>
            <input type="text" id="firstName" name="firstName" required>
            <br><br>
            
            <label for="lastName">Last Name:</label>
            <input type="text" id="lastName" name="lastName" required>
            <br><br>
            
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required>
            <br><br>
            
            <label for="phone">Phone:</label>
            <input type="tel" id="phone" name="phone">
            <br><br>
            
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" min="1" max="120">
        </fieldset>
        
        <!-- Preferences -->
        <fieldset>
            <legend>Contact Preferences</legend>
            
            <label>Preferred contact method:</label><br>
            <input type="radio" id="emailPref" name="contactMethod" value="email" checked>
            <label for="emailPref">Email</label><br>
            
            <input type="radio" id="phonePref" name="contactMethod" value="phone">
            <label for="phonePref">Phone</label><br><br>
            
            <label>Services interested in:</label><br>
            <input type="checkbox" id="webDev" name="services" value="web-development">
            <label for="webDev">Web Development</label><br>
            
            <input type="checkbox" id="design" name="services" value="design">
            <label for="design">Graphic Design</label><br>
            
            <input type="checkbox" id="seo" name="services" value="seo">
            <label for="seo">SEO Services</label><br><br>
            
            <label for="budget">Budget Range:</label>
            <select id="budget" name="budget">
                <option value="">Select budget</option>
                <option value="under-1000">Under $1,000</option>
                <option value="1000-5000">$1,000 - $5,000</option>
                <option value="5000-10000">$5,000 - $10,000</option>
                <option value="over-10000">Over $10,000</option>
            </select>
        </fieldset>
        
        <!-- Message -->
        <fieldset>
            <legend>Your Message</legend>
            
            <label for="subject">Subject:</label>
            <input type="text" id="subject" name="subject" required>
            <br><br>
            
            <label for="message">Message:</label><br>
            <textarea id="message" name="message" rows="5" cols="50" required placeholder="Please describe your project or inquiry..."></textarea>
        </fieldset>
        
        <!-- Submit -->
        <br>
        <input type="submit" value="Send Message">
        <input type="reset" value="Clear Form">
    </form>
</body>
</html>
```

---

## Lecture 9: Semantic HTML5 Elements

Semantic elements clearly describe their meaning to both browsers and developers.

### Why Use Semantic HTML?

1. **Better SEO** - Search engines understand your content better
2. **Accessibility** - Screen readers can navigate more easily  
3. **Code Clarity** - Makes code more readable and maintainable
4. **Future-proof** - Works better with new technologies

### Main Semantic Elements

#### Page Structure
```html
<header>
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
        </ul>
    </nav>
</header>

<main>
    <article>
        <header>
            <h1>Article Title</h1>
        </header>
        <section>
            <h2>Section Title</h2>
            <p>Content...</p>
        </section>
    </article>
    
    <aside>
        <h3>Related Links</h3>
    </aside>
</main>

<footer>
    <p>&copy; 2024 My Website</p>
</footer>
```

### Semantic Elements Explained

#### Structure Elements
- `<header>` - Top section, logo, navigation
- `<nav>` - Navigation links
- `<main>` - Main content area
- `<footer>` - Bottom section, copyright, contact info
- `<aside>` - Sidebar content, related info

#### Content Elements
- `<article>` - Standalone content (blog post, news article)
- `<section>` - Themed group of content
- `<figure>` - Images, diagrams, code snippets
- `<figcaption>` - Caption for figure

#### Text Elements
- `<time>` - Dates and times
- `<mark>` - Highlighted text
- `<details>` - Collapsible content
- `<summary>` - Summary for details

### Complete Semantic Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tech Blog - HTML5 Semantic Elements</title>
</head>
<body>
    <header>
        <h1>Tech Today Blog</h1>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#articles">Articles</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <article>
            <header>
                <h2>Understanding HTML5 Semantic Elements</h2>
                <p>Published by <strong>John Doe</strong> on 
                   <time datetime="2024-01-15">January 15, 2024</time>
                </p>
            </header>
            
            <section>
                <h3>What Are Semantic Elements?</h3>
                <p>Semantic elements are HTML tags that provide meaning to the content they contain. Unlike generic <code>div</code> or <code>span</code> elements, semantic elements describe their purpose.</p>
                
                <figure>
                    <img src="semantic-html.png" alt="HTML5 semantic elements diagram" width="400">
                    <figcaption>HTML5 semantic elements provide structure and meaning</figcaption>
                </figure>
            </section>
            
            <section>
                <h3>Benefits of Semantic HTML</h3>
                <p>Using semantic HTML provides several advantages:</p>
                <ul>
                    <li><mark>Better SEO</mark> - Search engines understand content structure</li>
                    <li><strong>Improved accessibility</strong> - Screen readers navigate more easily</li>
                    <li><em>Cleaner code</em> - More readable and maintainable</li>
                </ul>
                
                <details>
                    <summary>Click to see code example</summary>
                    <pre><code>&lt;article&gt;
    &lt;header&gt;
        &lt;h1&gt;Article Title&lt;/h1&gt;
    &lt;/header&gt;
    &lt;section&gt;
        &lt;p&gt;Article content...&lt;/p&gt;
    &lt;/section&gt;
&lt;/article&gt;</code></pre>
                </details>
            </section>
            
            <footer>
                <p>Tags: HTML5, Web Development, Semantic Web</p>
                <p>Last updated: <time datetime="2024-01-16">January 16, 2024</time></p>
            </footer>
        </article>
        
        <aside>
            <h3>Related Articles</h3>
            <ul>
                <li><a href="#">CSS Grid Layout Guide</a></li>
                <li><a href="#">JavaScript ES6 Features</a></li>
                <li><a href="#">Responsive Web Design Tips</a></li>
            </ul>
            
            <section>
                <h4>Newsletter</h4>
                <p>Subscribe for weekly web development tips!</p>
                <form>
                    <input type="email" placeholder="Your email">
                    <button type="submit">Subscribe</button>
                </form>
            </section>
        </aside>
    </main>
    
    <footer>
        <p>&copy; 2024 Tech Today Blog. All rights reserved.</p>
        <nav>
            <a href="#privacy">Privacy Policy</a> | 
            <a href="#terms">Terms of Service</a> | 
            <a href="#contact">Contact Us</a>
        </nav>
    </footer>
</body>
</html>
```

---

## Lecture 10: Best Practices and Next Steps

### HTML Best Practices

#### 1. Document Structure
```html
<!-- Always start with proper DOCTYPE -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Descriptive Page Title</title>
</head>
<body>
    <!-- Content here -->
</body>
</html>
```

#### 2. Semantic HTML
```html
<!-- Good: Semantic and meaningful -->
<header>
    <nav>
        <ul>
            <li><a href="#home">Home</a></li>
        </ul>
    </nav>
</header>
<main>
    <article>
        <h1>Article Title</h1>
        <p>Content...</p>
    </article>
</main>

<!-- Avoid: Non-semantic -->
<div id="header">
    <div id="navigation">
        <div class="menu-item">Home</div>
    </div>
</div>
```

#### 3. Accessibility Best Practices
```html
<!-- Always include alt text for images -->
<img src="chart.jpg" alt="Sales increased 25% in Q4 2023">

<!-- Use labels with form inputs -->
<label for="email">Email Address:</label>
<input type="email" id="email" name="email" required>

<!-- Provide meaningful link text -->
<a href="report.pdf">Download Q4 Sales Report (PDF, 2MB)</a>
<!-- Not: <a href="report.pdf">Click here</a> -->

<!-- Use headings in logical order -->
<h1>Main Title</h1>
<h2>Section Title</h2>
<h3>Subsection Title</h3>
<!-- Don't skip heading levels -->
```

#### 4. Code Organization
```html
<!-- Use consistent indentation -->
<article>
    <header>
        <h1>Title</h1>
        <p>Author info</p>
    </header>
    <section>
        <h2>Section Title</h2>
        <p>Content here...</p>
    </section>
</article>

<!-- Add helpful comments -->
<!-- Navigation Menu -->
<nav>
    <ul>
        <li><a href="#home">Home</a></li>
    </ul>
</nav>

<!-- Main Content Area -->
<main>
    <!-- Article content -->
</main>
```

#### 5. Performance Best Practices
```html
<!-- Optimize images -->
<img src="photo-small.jpg" alt="Team photo" 
     srcset="photo-small.jpg 320w, photo-medium.jpg 768w, photo-large.jpg 1024w"
     sizes="(max-width: 768px) 320px, (max-width: 1024px) 768px, 1024px">

<!-- Use appropriate input types -->
<input type="email" name="email">  <!-- Triggers email keyboard on mobile -->
<input type="tel" name="phone">    <!-- Triggers number pad -->
<input type="url" name="website">  <!-- Optimized for URLs -->
```

### Common HTML Mistakes to Avoid

#### 1. Structure Mistakes
```html
<!-- WRONG: Missing DOCTYPE -->
<html>
<head><title>Page</title></head>

<!-- CORRECT: Always include DOCTYPE -->
<!DOCTYPE html>
<html>
<head><title>Page</title></head>

<!-- WRONG: Unclosed tags -->
<p>Paragraph 1
<p>Paragraph 2

<!-- CORRECT: Properly closed tags -->
<p>Paragraph 1</p>
<p>Paragraph 2</p>
```

#### 2. Accessibility Mistakes
```html
<!-- WRONG: Missing alt text -->
<img src="important-chart.jpg">

<!-- CORRECT: Descriptive alt text -->
<img src="important-chart.jpg" alt="Website traffic increased 40% from January to March">

<!-- WRONG: Non-descriptive links -->
<a href="details.html">Click here</a>

<!-- CORRECT: Descriptive links -->
<a href="details.html">View product details</a>
```

#### 3. Form Mistakes
```html
<!-- WRONG: No labels -->
<input type="text" placeholder="Enter your name">

<!-- CORRECT: Proper labels -->
<label for="name">Name:</label>
<input type="text" id="name" name="name" placeholder="Enter your name">
```

### HTML Validation

Always validate your HTML to catch errors:
- Use the [W3C HTML Validator](https://validator.w3.org/)
- Check for proper nesting of elements
- Ensure all required attributes are present
- Verify semantic structure makes sense

### Tools and Resources

#### Development Tools
- **Text Editors**: VS Code, Sublime Text, Atom
- **Browser Dev Tools**: Chrome DevTools, Firefox Developer Tools
- **Online Editors**: CodePen, JSFiddle, Repl.it

#### Learning Resources
- **Documentation**: MDN Web Docs (developer.mozilla.org)
- **Practice**: freeCodeCamp, Codecademy
- **Communities**: Stack Overflow, Reddit r/webdev

### Next Steps After HTML

#### 1. CSS (Cascading Style Sheets)
```html
<!-- CSS makes your HTML look good -->
<head>
    <link rel="stylesheet" href="styles.css">
</head>
```

Learn CSS to:
- Style your HTML elements
- Create layouts and responsive designs
- Add animations and transitions
- Make your websites visually appealing

#### 2. JavaScript
```html
<!-- JavaScript makes your HTML interactive -->
<script src="script.js"></script>
```

Learn JavaScript to:
- Add interactivity to your pages
- Handle user events (clicks, form submissions)
- Manipulate HTML content dynamically
- Create modern web applications

#### 3. Modern Web Development
- **Frameworks**: React, Vue.js, Angular
- **Build Tools**: Webpack, Vite, Parcel
- **Version Control**: Git and GitHub
- **Responsive Design**: Mobile-first development

### Final Project Example

Here's a complete website example using everything we've learned:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Learn web development with our comprehensive HTML course">
    <title>WebDev Academy - Learn HTML</title>
</head>
<body>
    <!-- Header with Navigation -->
    <header>
        <div>
            <img src="logo.png" alt="WebDev Academy Logo" width="150">
            <h1>WebDev Academy</h1>
        </div>
        
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#courses">Courses</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>

    <!-- Main Content -->
    <main>
        <!-- Hero Section -->
        <section id="home">
            <h1>Learn Web Development</h1>
            <p>Master HTML, CSS, and JavaScript with our hands-on courses.</p>
            <a href="#courses">Start Learning Today</a>
        </section>

        <!-- Courses Section -->
        <section id="courses">
            <h2>Our Courses</h2>
            
            <article>
                <header>
                    <h3>HTML Fundamentals</h3>
                    <p><time>Duration: 4 weeks</time> | <strong>Beginner Level</strong></p>
                </header>
                
                <p>Learn the building blocks of web development with our comprehensive HTML course.</p>
                
                <h4>What You'll Learn:</h4>
                <ul>
                    <li>HTML document structure</li>
                    <li>Semantic elements</li>
                    <li>Forms and tables</li>
                    <li>Best practices</li>
                </ul>
                
                <footer>
                    <p>Price: <mark>$99</mark> | <a href="#enroll-html">Enroll Now</a></p>
                </footer>
            </article>

            <article>
                <header>
                    <h3>CSS Styling</h3>
                    <p><time>Duration: 6 weeks</time> | <strong>Intermediate Level</strong></p>
                </header>
                
                <p>Transform your HTML with beautiful styling and responsive layouts.</p>
                
                <details>
                    <summary>Course Outline</summary>
                    <ol>
                        <li>CSS Basics</li>
                        <li>Layout Techniques</li>
                        <li>Flexbox and Grid</li>
                        <li>Responsive Design</li>
                        <li>Animations</li>
                    </ol>
                </details>
                
                <footer>
                    <p>Price: <mark>$149</mark> | <a href="#enroll-css">Enroll Now</a></p>
                </footer>
            </article>
        </section>

        <!-- Student Testimonials -->
        <section>
            <h2>Student Success Stories</h2>
            
            <figure>
                <blockquote>
                    <p>"The HTML course was perfect for beginners. I went from knowing nothing to building my first website in just one month!"</p>
                </blockquote>
                <figcaption>
                    - Sarah Johnson, <cite>Software Developer</cite>
                </figcaption>
            </figure>
            
            <figure>
                <blockquote>
                    <p>"Excellent course structure and hands-on projects. The instructors are knowledgeable and supportive."</p>
                </blockquote>
                <figcaption>
                    - Mike Chen, <cite>Freelance Web Designer</cite>
                </figcaption>
            </figure>
        </section>

        <!-- Contact Form -->
        <section id="contact">
            <h2>Get in Touch</h2>
            
            <form action="/contact" method="POST">
                <fieldset>
                    <legend>Contact Information</legend>
                    
                    <label for="name">Full Name:</label>
                    <input type="text" id="name" name="name" required>
                    
                    <label for="email">Email Address:</label>
                    <input type="email" id="email" name="email" required>
                    
                    <label for="phone">Phone Number:</label>
                    <input type="tel" id="phone" name="phone">
                </fieldset>
                
                <fieldset>
                    <legend>Inquiry Type</legend>
                    
                    <input type="radio" id="course-info" name="inquiry" value="course-info" checked>
                    <label for="course-info">Course Information</label>
                    
                    <input type="radio" id="enrollment" name="inquiry" value="enrollment">
                    <label for="enrollment">Enrollment</label>
                    
                    <input type="radio" id="other" name="inquiry" value="other">
                    <label for="other">Other</label>
                </fieldset>
                
                <label for="message">Message:</label>
                <textarea id="message" name="message" rows="5" required 
                         placeholder="Tell us how we can help you..."></textarea>
                
                <button type="submit">Send Message</button>
                <button type="reset">Clear Form</button>
            </form>
        </section>
    </main>

    <!-- Sidebar -->
    <aside>
        <section>
            <h3>Quick Links</h3>
            <ul>
                <li><a href="#faq">FAQ</a></li>
                <li><a href="#schedule">Class Schedule</a></li>
                <li><a href="#resources">Free Resources</a></li>
                <li><a href="#blog">Blog</a></li>
            </ul>
        </section>
        
        <section>
            <h3>Upcoming Events</h3>
            <article>
                <h4>Free HTML Workshop</h4>
                <p><time datetime="2024-02-15">February 15, 2024</time></p>
                <p>Join us for a free 2-hour introduction to HTML basics.</p>
                <a href="#workshop">Register Now</a>
            </article>
        </section>
    </aside>

    <!-- Footer -->
    <footer>
        <section>
            <h3>WebDev Academy</h3>
            <p>Empowering the next generation of web developers</p>
            
            <address>
                <p>123 Tech Street<br>
                San Francisco, CA 94102<br>
                Phone: <a href="tel:+1555123456">(555) 123-456</a><br>
                Email: <a href="mailto:info@webdevacademy.com">info@webdevacademy.com</a></p>
            </address>
        </section>
        
        <section>
            <h4>Follow Us</h4>
            <ul>
                <li><a href="https://twitter.com/webdevacademy">Twitter</a></li>
                <li><a href="https://linkedin.com/company/webdevacademy">LinkedIn</a></li>
                <li><a href="https://github.com/webdevacademy">GitHub</a></li>
            </ul>
        </section>
        
        <section>
            <p><small>&copy; 2024 WebDev Academy. All rights reserved.</small></p>
            <nav>
                <a href="#privacy">Privacy Policy</a> | 
                <a href="#terms">Terms of Service</a> | 
                <a href="#accessibility">Accessibility</a>
            </nav>
        </section>
    </footer>
</body>
</html>
```

### Summary

Congratulations! You've learned:

1. **HTML Basics** - Structure, tags, and elements
2. **Document Structure** - Proper HTML5 document setup
3. **Text Formatting** - Headings, paragraphs, and emphasis
4. **Lists** - Ordered, unordered, and description lists
5. **Links** - Navigation and connectivity
6. **Images** - Adding visual content
7. **Tables** - Displaying structured data
8. **Forms** - User input and interaction
9. **Semantic HTML** - Meaningful, accessible markup
10. **Best Practices** - Professional development standards

### Your HTML Journey Continues

HTML is just the beginning of your web development journey. With this solid foundation, you're ready to:

- Style your pages with CSS
- Add interactivity with JavaScript
- Build responsive, modern websites
- Explore advanced frameworks and tools

Remember: **Practice makes perfect**. Build projects, experiment with code, and never stop learning!

### Quick Reference

#### Essential HTML Tags
```html
<!-- Document Structure -->
<!DOCTYPE html>, <html>, <head>, <body>

<!-- Text Content -->
<h1> to <h6>, <p>, <strong>, <em>, <br>, <hr>

<!-- Lists -->
<ul>, <ol>, <li>, <dl>, <dt>, <dd>

<!-- Links and Media -->
<a>, <img>, <figure>, <figcaption>

<!-- Tables -->
<table>, <thead>, <tbody>, <tr>, <th>, <td>

<!-- Forms -->
<form>, <input>, <label>, <select>, <textarea>, <button>

<!-- Semantic Elements -->
<header>, <nav>, <main>, <section>, <article>, <aside>, <footer>
```

Good luck with your web development journey!