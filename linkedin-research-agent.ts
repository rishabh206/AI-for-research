/**
 * Cozmo AI — LinkedIn SDR Research Agent
 *
 * Reads your LinkedIn connections CSV export and acts as a B2B SDR for Cozmo AI.
 * For each connection it:
 *   1. Researches the prospect (role, company, likely pain points)
 *   2. Identifies AI customer support automation opportunities
 *   3. Qualifies using BANT (Budget, Authority, Need, Timeline)
 *   4. Drafts a personalized LinkedIn DM + cold email
 *
 * Usage:
 *   npm run research -- Connections.csv
 *
 * To export your LinkedIn connections:
 *   LinkedIn → Me → Settings & Privacy → Data Privacy → Get a copy of your data
 *   → Select "Connections" → Request archive → Download Connections.csv
 */

import Anthropic from "@anthropic-ai/sdk";
import fs from "fs";
import path from "path";

const client = new Anthropic();

// ─── Cozmo AI SDR context ─────────────────────────────────────────────────────

const COZMO_SYSTEM_PROMPT = `You are a senior B2B SDR (Sales Development Representative) at Cozmo AI.

ABOUT COZMO AI:
Cozmo AI is an AI-powered customer support automation platform for SMBs.
It deflects repetitive support tickets, auto-responds to common queries, and
routes complex issues to humans — cutting support costs and response times.

IDEAL CUSTOMER PROFILE (ICP):
- Company size: 10–200 employees (SMB)
- Roles: Head of Support, VP Customer Success, Head of CX, COO, Founder/CEO
- Pain: High ticket volume, slow response times, support team scaling issues
- Tech: SaaS, e-commerce, fintech, marketplace, or subscription businesses
- Budget: <$10K ACV; short sales cycle (<30 days); self-serve or low-touch

YOUR GOALS FOR EVERY PROSPECT:
1. RESEARCH — Infer their likely support challenges from role + company
2. AUTOMATION OPPORTUNITIES — Identify specific workflows Cozmo AI could automate
3. BANT QUALIFICATION — Score Budget / Authority / Need / Timeline (1–3 each)
4. OUTREACH — Write a personalized LinkedIn DM and a cold email

TONE: Concise, human, consultative. Never pitch immediately. Lead with insight.
BANT SCORING: 1 = weak signal, 2 = moderate, 3 = strong signal based on available info.`;

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

interface BANTScore {
  budget: number;     // 1–3
  authority: number;  // 1–3
  need: number;       // 1–3
  timeline: number;   // 1–3
  total: number;      // sum, 4–12
  notes: string;
}

interface ProspectIntelligence extends Connection {
  // ICP fit
  icpScore: number;         // 1–10
  tier: "hot" | "warm" | "cold";

  // Research
  likelyPainPoints: string;
  automationOpportunities: string;

  // BANT
  bant: BANTScore;

  // Outreach
  linkedinDM: string;
  coldEmailSubject: string;
  coldEmailBody: string;
}

// ─── CSV Parsing ──────────────────────────────────────────────────────────────

function parseCSV(filePath: string): Connection[] {
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split(/\r?\n/).filter((l: string) => l.trim());

  if (lines.length < 2) throw new Error("CSV file appears empty.");

  const headerIndex = lines.findIndex((l: string) =>
    l.toLowerCase().includes("first name")
  );
  if (headerIndex === -1) {
    throw new Error(
      'Could not find "First Name" header. Export Connections from LinkedIn.'
    );
  }

  const headers = parseCSVRow(lines[headerIndex]).map((h: string) =>
    h.toLowerCase().replace(/\s+/g, "")
  );

  const connections: Connection[] = [];
  for (let i = headerIndex + 1; i < lines.length; i++) {
    const values = parseCSVRow(lines[i]);
    if (values.length < 3) continue;

    const row: Record<string, string> = {};
    headers.forEach((h: string, idx: number) => {
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
      if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
      else inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      values.push(current); current = "";
    } else {
      current += ch;
    }
  }
  values.push(current);
  return values;
}

// ─── SDR Analysis ─────────────────────────────────────────────────────────────

function buildProspectLine(c: Connection, idx: number): string {
  const parts: string[] = [`${idx + 1}.`];
  parts.push(`${c.firstName} ${c.lastName}`.trim() || "(unknown)");
  if (c.position) parts.push(`| ${c.position}`);
  if (c.company) parts.push(`@ ${c.company}`);
  return parts.join(" ");
}

async function analyzeProspectBatch(
  connections: Connection[]
): Promise<ProspectIntelligence[]> {
  const prospectList = connections
    .map((c, i) => buildProspectLine(c, i))
    .join("\n");

  const userPrompt = `Analyze these LinkedIn connections as potential Cozmo AI customers.
For each prospect, apply your SDR expertise and return a JSON array.

PROSPECTS:
${prospectList}

Return a JSON array where each element has exactly these fields:
{
  "index": <1-based integer matching the list>,
  "icpScore": <integer 1–10; 8+ = strong ICP fit>,
  "tier": <"hot" if icpScore>=7, "warm" if 4–6, "cold" if <=3>,
  "likelyPainPoints": <1–2 sentences on their probable support challenges>,
  "automationOpportunities": <1–2 specific Cozmo AI use cases for this person's company>,
  "bant": {
    "budget": <1–3>,
    "authority": <1–3>,
    "need": <1–3>,
    "timeline": <1–3>,
    "total": <sum of above>,
    "notes": <1 sentence on BANT rationale>
  },
  "linkedinDM": <a short (3–4 sentence) personalized LinkedIn DM — insight-led, no hard pitch>,
  "coldEmailSubject": <a sharp subject line under 50 chars>,
  "coldEmailBody": <a 5–7 sentence cold email: hook → pain → solution → CTA. Sign as "Rishabh, Cozmo AI">
}

Return ONLY the JSON array. No extra text.`;

  const response = await client.messages.create({
    model: "claude-opus-4-8",
    max_tokens: 8192,
    thinking: { type: "adaptive" },
    system: COZMO_SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
  });

  const textBlock = response.content.find((b) => b.type === "text");
  if (!textBlock || textBlock.type !== "text") {
    throw new Error("No text response from Claude");
  }

  let jsonText = textBlock.text.trim();
  jsonText = jsonText.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");

  type RawResult = {
    index: number;
    icpScore: number;
    tier: string;
    likelyPainPoints: string;
    automationOpportunities: string;
    bant: {
      budget: number;
      authority: number;
      need: number;
      timeline: number;
      total: number;
      notes: string;
    };
    linkedinDM: string;
    coldEmailSubject: string;
    coldEmailBody: string;
  };

  const parsed: RawResult[] = JSON.parse(jsonText);

  return parsed.map((r) => {
    const conn = connections[r.index - 1];
    return {
      ...conn,
      icpScore: r.icpScore,
      tier: r.tier as ProspectIntelligence["tier"],
      likelyPainPoints: r.likelyPainPoints,
      automationOpportunities: r.automationOpportunities,
      bant: r.bant,
      linkedinDM: r.linkedinDM,
      coldEmailSubject: r.coldEmailSubject,
      coldEmailBody: r.coldEmailBody,
    };
  });
}

// ─── Output ───────────────────────────────────────────────────────────────────

function writeResultsCSV(results: ProspectIntelligence[], outputPath: string) {
  const header = [
    "ICP Score", "Tier", "BANT Total", "Budget", "Authority", "Need", "Timeline",
    "First Name", "Last Name", "Position", "Company", "Email", "Connected On", "Profile URL",
    "Likely Pain Points", "Automation Opportunities", "BANT Notes",
    "LinkedIn DM", "Cold Email Subject", "Cold Email Body",
  ].join(",");

  const rows = results.map((r) => {
    const cells = [
      r.icpScore, r.tier, r.bant.total,
      r.bant.budget, r.bant.authority, r.bant.need, r.bant.timeline,
      r.firstName, r.lastName, r.position, r.company,
      r.emailAddress, r.connectedOn, r.profileUrl,
      r.likelyPainPoints, r.automationOpportunities, r.bant.notes,
      r.linkedinDM, r.coldEmailSubject, r.coldEmailBody,
    ].map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`);
    return cells.join(",");
  });

  fs.writeFileSync(outputPath, [header, ...rows].join("\n"), "utf-8");
}

function printReport(results: ProspectIntelligence[]) {
  const hot = results.filter((r) => r.tier === "hot");
  const warm = results.filter((r) => r.tier === "warm");
  const cold = results.filter((r) => r.tier === "cold");

  const avgBant = (arr: ProspectIntelligence[]) =>
    arr.length ? (arr.reduce((s, r) => s + r.bant.total, 0) / arr.length).toFixed(1) : "—";

  console.log("\n" + "═".repeat(68));
  console.log("  COZMO AI SDR REPORT");
  console.log("═".repeat(68));
  console.log(
    `  Analyzed: ${results.length}  |  🔥 Hot: ${hot.length}  |  🌡️  Warm: ${warm.length}  |  ❄️  Cold: ${cold.length}`
  );
  console.log(
    `  Avg BANT — Hot: ${avgBant(hot)}/12  |  Warm: ${avgBant(warm)}/12`
  );
  console.log("═".repeat(68));

  if (hot.length > 0) {
    console.log("\n🔥  HOT PROSPECTS  (ICP score 7–10)\n");
    hot.slice(0, 8).forEach((r) => {
      console.log(`  [${ r.icpScore}/10 | BANT ${r.bant.total}/12]  ${r.firstName} ${r.lastName}`);
      console.log(`  ${r.position} @ ${r.company}`);
      console.log(`  Pain:        ${r.likelyPainPoints}`);
      console.log(`  Automation:  ${r.automationOpportunities}`);
      console.log(`  BANT:        ${r.bant.notes}`);
      console.log(`\n  LinkedIn DM:\n  ${r.linkedinDM.replace(/\n/g, "\n  ")}`);
      console.log(`\n  Email — ${r.coldEmailSubject}`);
      console.log(`  ${r.coldEmailBody.replace(/\n/g, "\n  ")}`);
      console.log("\n" + "─".repeat(68));
    });
    if (hot.length > 8) {
      console.log(`  + ${hot.length - 8} more hot prospects in the CSV.\n`);
    }
  }

  if (warm.length > 0) {
    console.log("\n🌡️   TOP WARM PROSPECTS  (ICP score 4–6)\n");
    warm
      .sort((a, b) => b.bant.total - a.bant.total)
      .slice(0, 4)
      .forEach((r) => {
        console.log(`  [${r.icpScore}/10 | BANT ${r.bant.total}/12]  ${r.firstName} ${r.lastName} — ${r.position} @ ${r.company}`);
        console.log(`  ${r.likelyPainPoints}\n`);
      });
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const [, , csvArg] = process.argv;

  if (!csvArg) {
    console.error("Usage: npm run research -- Connections.csv");
    console.error(
      "\nExport your connections:\n" +
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

  console.log("\nCozmo AI SDR Agent starting...");
  console.log("Parsing connections...");
  const connections = parseCSV(csvPath);
  console.log(`Found ${connections.length} connections.\n`);

  // Filter out connections with no useful data
  const actionable = connections.filter((c) => c.position || c.company);
  const skipped = connections.length - actionable.length;
  if (skipped > 0) {
    console.log(`Skipping ${skipped} connections with no role/company data.\n`);
  }

  const BATCH_SIZE = 15; // smaller batches for richer per-prospect output
  const batches: Connection[][] = [];
  for (let i = 0; i < actionable.length; i += BATCH_SIZE) {
    batches.push(actionable.slice(i, i + BATCH_SIZE));
  }

  console.log(`Analyzing ${actionable.length} prospects in ${batches.length} batches...\n`);

  const allResults: ProspectIntelligence[] = [];
  for (let i = 0; i < batches.length; i++) {
    process.stdout.write(
      `  [${i + 1}/${batches.length}] ${batches[i].length} prospects... `
    );
    try {
      const results = await analyzeProspectBatch(batches[i]);
      allResults.push(...results);
      console.log("✓");
    } catch (err) {
      console.log("error — skipping");
      console.error(`    ${err instanceof Error ? err.message : err}`);
    }
  }

  allResults.sort((a, b) => b.icpScore - a.icpScore || b.bant.total - a.bant.total);

  const outputPath = csvPath.replace(/\.csv$/i, "") + "_sdr_intelligence.csv";
  writeResultsCSV(allResults, outputPath);

  printReport(allResults);

  console.log(`\n✅  Full SDR intelligence saved to: ${outputPath}\n`);
}

main().catch((err) => {
  console.error("Fatal:", err instanceof Error ? err.message : err);
  process.exit(1);
});
