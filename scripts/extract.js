import {
  readFileSync,
  writeFileSync,
  readdirSync,
  statSync,
  rmSync,
  mkdirSync,
  existsSync,
} from "fs";
import { unified } from "unified";
import parse from "remark-parse";
import stringify from "remark-stringify";
import { join } from "path";
import { v4 as uuidv4 } from "uuid";

function analyzeHeadingLevels(tree) {
  const levels = new Set();
  tree.children.forEach((node) => {
    if (node.type === "heading") {
      levels.add(node.depth);
    }
  });
  return Array.from(levels).sort((a, b) => a - b);
}

function findOptimalSplitLevel(tree) {
  const levels = analyzeHeadingLevels(tree);
  if (levels.length === 0) return null;
  if (levels.length === 1) return levels[0];

  const middleIndex = Math.floor(levels.length / 2);
  return levels[middleIndex];
}

function getContextNodes(nodes, currentIndex, contextSize = 1) {
  const start = Math.max(0, currentIndex - contextSize);
  const end = Math.min(nodes.length, currentIndex + contextSize + 1);
  return nodes.slice(start, end);
}

function splitByOptimalLevel(input, contextSize = 1) {
  const processor = unified().use(parse).use(stringify);

  const tree = processor.parse(input);
  const optimalLevel = findOptimalSplitLevel(tree);

  if (!optimalLevel) {
    return [
      {
        title: "Document",
        content: input,
        context: {
          before: "",
          after: "",
        },
      },
    ];
  }

  const sections = [];
  let currentSection = [];
  let currentTitle = "";
  let currentIndex = 0;

  tree.children.forEach((node, index) => {
    if (node.type === "heading" && node.depth === optimalLevel) {
      if (currentSection.length > 0) {
        const contextNodes = getContextNodes(
          tree.children,
          currentIndex,
          contextSize
        );
        sections.push({
          title: currentTitle,
          content: processor.stringify({
            type: "root",
            children: currentSection,
          }),
          context: {
            before: processor.stringify({
              type: "root",
              children: contextNodes.slice(0, contextSize),
            }),
            after: processor.stringify({
              type: "root",
              children: contextNodes.slice(contextSize + 1),
            }),
          },
        });
        currentSection = [];
      }
      currentTitle = node.children.map((child) => child.value).join("");
      currentIndex = index;
    }
    currentSection.push(node);
  });

  if (currentSection.length > 0) {
    const contextNodes = getContextNodes(
      tree.children,
      currentIndex,
      contextSize
    );
    sections.push({
      title: currentTitle,
      content: processor.stringify({ type: "root", children: currentSection }),
      context: {
        before: processor.stringify({
          type: "root",
          children: contextNodes.slice(0, contextSize),
        }),
        after: processor.stringify({
          type: "root",
          children: contextNodes.slice(contextSize + 1),
        }),
      },
    });
  }

  return sections;
}

const contextSize = 2;

if (existsSync("splited_files")) {
  rmSync("splited_files", { recursive: true, force: true });
}

mkdirSync("splited_files");

processFolder(process.argv[2]);

function processFolder(folderPath) {
  const files = readdirSync(folderPath);
  files.forEach((file) => {
    if (statSync(join(folderPath, file)).isDirectory()) {
      processFolder(join(folderPath, file));
    } else if (file.endsWith(".md")) {
      const filePath = join(folderPath, file);
      const input = readFileSync(filePath, "utf8");
      const sections = splitByOptimalLevel(input, contextSize);
      sections.forEach((section, index) => {
        if (section.title) {
          const baseFilename = section.title
            .replace(/[^a-z0-9]/gi, "_")
            .toLowerCase();

          const mainFilename = `section_${baseFilename}.md`;
          const uuid = uuidv4();
          writeFileSync(
            join("splited_files", uuid + "_" + mainFilename),
            section.content
          );
          console.log(`Created main content: ${mainFilename}`);

          const contextFilename = `context_${baseFilename}.md`;
          const contextContent = `# Context Information for "${section.title}"
      
      This section is part of a larger document. Here is the context information for LLM processing:
      
      ## Previous Content
      ${section.context.before || "No previous content available."}
      
      ## Next Content
      ${section.context.after || "No next content available."}
      
      This context information helps understand the relationship between this section and its surrounding content.`;

          writeFileSync(
            join("splited_files", uuid + "_" + contextFilename),
            contextContent
          );
          console.log(`Created context info: ${contextFilename}`);
        }
      });
    }
  });
}
