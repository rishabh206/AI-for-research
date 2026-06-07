/**
 * LinkedIn Connections Research Agent
 *
 * Reads your LinkedIn connections CSV export and uses Claude to identify
 * and score potential customers.
 *
 * Usage:
 *   npx ts-node linkedin-research-agent.ts <connections.csv> [product-description]
 *
 * To export your LinkedIn connections:
 *   LinkedIn → Me → Settings & Privacy → Data Privacy → Get a copy of your data
 *   → Select "Connections" → Request archive → Download Connections.csv
 */

import Anthropic from "@anthropic-ai/sdk";
import fs from "fs";
import path from "path";
import readline from "readline";

const client = new Anthropic();

// ─── Types ────────────────────────────────────────────────────────────────────

interface Connection {
  firstName: string;
  lastName: string;
  emailAddress: string;
  company: string;
  position: string;
  connectedOn: string;
  profileUrl: string;
}

interface ScoredConnection extends Connection {
  score: number; // 1–10
  tier: "hot" | "warm" | "cold";
  reasoning: string;
  suggestedApproach: string;
}

interface AnalysisBatch {
  connections: Connection[];
  results: ScoredConnection[];
}

// ─── CSV Parsing ──────────────────────────────────────────────────────────────

function parseCSV(filePath: string): Connection[] {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split(/\r?\n/).filter((l) => l.trim());

  if (lines.length < 2) {
    throw new Error("CSV file appears empty or has no data rows.");
  }

  // LinkedIn CSV has a few header lines before the actual data; find the real header
  const headerIndex = lines.findIndex((l) =>
    l.toLowerCase().includes("first name")
  );
  if (headerIndex === -1) {
    throw new Error(
      'Could not find header row with "First Name" in the CSV. ' +
        "Make sure you exported Connections from LinkedIn."
    );
  }

  const headers = parseCSVRow(lines[headerIndex]).map((h) =>
    h.toLowerCase().replace(/\s+/g, "")
  );

  const connections: Connection[] = [];
  for (let i = headerIndex + 1; i < lines.length; i++) {
    const values = parseCSVRow(lines[i]);
    if (values.length < 3) continue;

    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = (values[idx] ?? "").trim();
    });

    connections.push({
      firstName: row["firstname"] ?? "",
      lastName: row["lastname"] ?? "",
      emailAddress: row["emailaddress"] ?? "",
      company: row["company"] ?? "",
      position: row["position"] ?? "",
      connectedOn: row["connectedon"] ?? "",
      profileUrl: row["profileurl"] ?? row["url"] ?? "",
    });
  }

  return connections.filter((c) => c.firstName || c.lastName);
}

function parseCSVRow(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  values.push(current);
  return values;
}

// ─── Claude Analysis ──────────────────────────────────────────────────────────

function buildConnectionSummary(c: Connection): string {
  const parts = [`${c.firstName} ${c.lastName}`.trim()];
  if (c.position) parts.push(`Position: ${c.position}`);
  if (c.company) parts.push(`Company: ${c.company}`);
  if (c.connectedOn) parts.push(`Connected: ${c.connectedOn}`);
  return parts.join(" | ");
}

async function analyzeConnectionBatch(
  connections: Connection[],
  productDescription: string
): Promise<ScoredConnection[]> {
  const connectionList = connections
    .map((c, i) => `${i + 1}. ${buildConnectionSummary(c)}`)
    .join("\n");

  const prompt = `You are a sales intelligence analyst. Evaluate the following LinkedIn connections as potential customers for this product/service:

PRODUCT/SERVICE:
${productDescription}

LINKEDIN CONNECTIONS TO EVALUATE:
${connectionList}

For each connection, return a JSON array. Each element must have:
- index: (number, 1-based, matching the list above)
- score: (integer 1–10, where 10 = perfect customer fit)
- tier: ("hot" if score >= 7, "warm" if 4–6, "cold" if <= 3)
- reasoning: (1–2 sentences explaining the score based on their role/company)
- suggestedApproach: (1 sentence on how to reach out)

Return ONLY the JSON array, no other text.`;

  const response = await client.messages.create({
    model: "claude-opus-4-8",
    max_tokens: 4096,
    thinking: { type: "adaptive" },
    messages: [{ role: "user", content: prompt }],
  });

  const textBlock = response.content.find((b) => b.type === "text");
  if (!textBlock || textBlock.type !== "text") {
    throw new Error("No text response from Claude");
  }

  let jsonText = textBlock.text.trim();
  // Strip markdown code fences if present
  jsonText = jsonText.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");

  const parsed: Array<{
    index: number;
    score: number;
    tier: string;
    reasoning: string;
    suggestedApproach: string;
  }> = JSON.parse(jsonText);

  return parsed.map((r) => {
    const conn = connections[r.index - 1];
    return {
      ...conn,
      score: r.score,
      tier: r.tier as ScoredConnection["tier"],
      reasoning: r.reasoning,
      suggestedApproach: r.suggestedApproach,
    };
  });
}

// ─── Output ───────────────────────────────────────────────────────────────────

function writeResultsCSV(results: ScoredConnection[], outputPath: string) {
  const header =
    "Score,Tier,First Name,Last Name,Position,Company,Email,Connected On,Profile URL,Reasoning,Suggested Approach";
  const rows = results.map((r) => {
    const cells = [
      r.score,
      r.tier,
      r.firstName,
      r.lastName,
      r.position,
      r.company,
      r.emailAddress,
      r.connectedOn,
      r.profileUrl,
      r.reasoning,
      r.suggestedApproach,
    ].map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`);
    return cells.join(",");
  });
  fs.writeFileSync(outputPath, [header, ...rows].join("\n"), "utf-8");
}

function printSummary(results: ScoredConnection[]) {
  const hot = results.filter((r) => r.tier === "hot");
  const warm = results.filter((r) => r.tier === "warm");
  const cold = results.filter((r) => r.tier === "cold");

  console.log("\n" + "═".repeat(60));
  console.log("  RESULTS SUMMARY");
  console.log("═".repeat(60));
  console.log(
    `  Total analyzed: ${results.length}  |  🔥 Hot: ${hot.length}  |  🌡️  Warm: ${warm.length}  |  ❄️  Cold: ${cold.length}`
  );
  console.log("═".repeat(60));

  if (hot.length > 0) {
    console.log("\n🔥 HOT LEADS (score 7–10)\n");
    hot.slice(0, 10).forEach((r) => {
      console.log(
        `  [${r.score}/10] ${r.firstName} ${r.lastName} — ${r.position} @ ${r.company}`
      );
      console.log(`         ${r.reasoning}`);
      console.log(`         → ${r.suggestedApproach}\n`);
    });
    if (hot.length > 10) {
      console.log(`  ... and ${hot.length - 10} more hot leads in the CSV.\n`);
    }
  }

  if (warm.length > 0) {
    console.log("🌡️  TOP WARM LEADS (score 4–6)\n");
    warm
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .forEach((r) => {
        console.log(
          `  [${r.score}/10] ${r.firstName} ${r.lastName} — ${r.position} @ ${r.company}`
        );
        console.log(`         ${r.reasoning}\n`);
      });
  }
}

// ─── Interactive product description ─────────────────────────────────────────

async function promptForProduct(): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) => {
    rl.question(
      "\nDescribe your product/service (who it's for, what it does, who's the ideal customer):\n> ",
      (answer) => {
        rl.close();
        resolve(answer.trim());
      }
    );
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const [, , csvArg, ...rest] = process.argv;

  if (!csvArg) {
    console.error(
      "Usage: npx ts-node linkedin-research-agent.ts <connections.csv> [product-description]"
    );
    console.error(
      "\nTo get your LinkedIn connections CSV:\n" +
        "  LinkedIn → Me → Settings & Privacy → Data Privacy\n" +
        "  → Get a copy of your data → Connections → Request archive"
    );
    process.exit(1);
  }

  const csvPath = path.resolve(csvArg);
  if (!fs.existsSync(csvPath)) {
    console.error(`File not found: ${csvPath}`);
    process.exit(1);
  }

  let productDescription = rest.join(" ").trim();
  if (!productDescription) {
    productDescription = await promptForProduct();
  }
  if (!productDescription) {
    console.error("Product description is required.");
    process.exit(1);
  }

  console.log("\nParsing connections...");
  const connections = parseCSV(csvPath);
  console.log(`Found ${connections.length} connections.`);

  const BATCH_SIZE = 25;
  const batches: Connection[][] = [];
  for (let i = 0; i < connections.length; i += BATCH_SIZE) {
    batches.push(connections.slice(i, i + BATCH_SIZE));
  }

  console.log(
    `Analyzing in ${batches.length} batches of up to ${BATCH_SIZE}...\n`
  );

  const allResults: ScoredConnection[] = [];
  for (let i = 0; i < batches.length; i++) {
    process.stdout.write(
      `  Batch ${i + 1}/${batches.length} (${batches[i].length} connections)... `
    );
    try {
      const results = await analyzeConnectionBatch(
        batches[i],
        productDescription
      );
      allResults.push(...results);
      console.log("done");
    } catch (err) {
      console.log("error — skipping batch");
      console.error(`    ${err instanceof Error ? err.message : err}`);
    }
  }

  // Sort by score descending
  allResults.sort((a, b) => b.score - a.score);

  const outputPath = csvPath.replace(/\.csv$/i, "") + "_potential_customers.csv";
  writeResultsCSV(allResults, outputPath);

  printSummary(allResults);

  console.log(`\n✅ Full results saved to: ${outputPath}\n`);
}

main().catch((err) => {
  console.error("Fatal:", err instanceof Error ? err.message : err);
  process.exit(1);
});
