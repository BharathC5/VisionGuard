document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("vgForm");
  const testedFields = document.getElementById("testedFields");
  const lastVisitIfNoFamily = document.getElementById("lastVisitIfNoFamily");
  const recommendBtn = document.getElementById("recommendBtn");
  const resetBtn = document.getElementById("resetBtn");
  const output = document.getElementById("output");

  // --- Toggle "tested" conditional fields ---
  const testedRadios = document.querySelectorAll('input[name="tested"]');
  testedRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.value === "yes" && radio.checked) {
        testedFields.style.display = "block";
        testedFields.setAttribute("aria-hidden", "false");
      } else if (radio.value === "no" && radio.checked) {
        testedFields.style.display = "none";
        testedFields.setAttribute("aria-hidden", "true");
      }
    });
  });

  // --- Toggle "last doctor visit" when family history = no ---
  const familyRadios = document.querySelectorAll('input[name="family"]');
  familyRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.value === "no" && radio.checked) {
        lastVisitIfNoFamily.style.display = "block";
        lastVisitIfNoFamily.setAttribute("aria-hidden", "false");
      } else if (radio.value === "yes" && radio.checked) {
        lastVisitIfNoFamily.style.display = "none";
        lastVisitIfNoFamily.setAttribute("aria-hidden", "true");
      }
    });
  });

  // --- Read form data ---
  function getFormData() {
    const age = parseInt(document.getElementById("age").value, 10);
    const stageInput = document.getElementById("stageInput");
    const stageRaw = stageInput.value === "" ? null : Number(stageInput.value);
    const tested = (
      document.querySelector('input[name="tested"]:checked') || {}
    ).value;
    const family = (
      document.querySelector('input[name="family"]:checked') || {}
    ).value;

    const lastTestDate = document.getElementById("lastTestDate").value || null;
    const testReadingRaw = document.getElementById("testReading").value || null;
    const lastDoctorVisit =
      document.getElementById("lastDoctorVisit").value || null;

    // Try to parse numeric part of test reading if user wrote "180 mg/dL"
    let testReadingNum = null;
    if (testReadingRaw) {
      const match = testReadingRaw.match(/(\d+(\.\d+)?)/);
      if (match) {
        testReadingNum = parseFloat(match[1]);
      }
    }

    return {
      age,
      stage: stageRaw,
      tested,
      family,
      lastTestDate,
      testReadingRaw,
      testReadingNum,
      lastDoctorVisit,
    };
  }

  // --- Neuro-fuzzy style risk computation ---
  function computeFuzzyRisk(data) {
    // 0 <= risk <= 1
    let risk = 0;

    // 1) Stage contribution (strongest)
    // If no stage, treat as 0 (screening only)
    const stage = typeof data.stage === "number" ? data.stage : 0;
    // normalize 0–4 to 0–1
    const stageScore = Math.min(Math.max(stage / 4, 0), 1);
    risk += 0.55 * stageScore; // stage dominates more than half of risk

    // 2) Age fuzziness
    // young: < 35, mid: 35–55, older: > 55
    if (!Number.isNaN(data.age)) {
      if (data.age <= 35) {
        risk += 0.05; // base risk
      } else if (data.age <= 55) {
        risk += 0.15; // moderate age risk
      } else {
        risk += 0.25; // older age → higher risk contribution
      }
    }

    // 3) Family history
    if (data.family === "yes") {
      risk += 0.1;
    }

    // 4) Tested + high glucose reading
    if (data.tested === "yes") {
      risk += 0.05; // tested itself adds a bit (suspicion / known risk)
      if (typeof data.testReadingNum === "number") {
        if (data.testReadingNum >= 200) {
          risk += 0.2; // very high reading
        } else if (data.testReadingNum >= 160) {
          risk += 0.12;
        } else if (data.testReadingNum >= 126) {
          risk += 0.08;
        }
      }
    }

    // clamp final risk
    risk = Math.min(Math.max(risk, 0), 1);

    // Label buckets
    let label = "Low";
    let cssClass = "risk-low";

    if (risk < 0.25) {
      label = "Low";
      cssClass = "risk-low";
    } else if (risk < 0.5) {
      label = "Moderate";
      cssClass = "risk-moderate";
    } else if (risk < 0.75) {
      label = "High";
      cssClass = "risk-high";
    } else {
      label = "Urgent";
      cssClass = "risk-urgent";
    }

    return { risk, label, cssClass };
  }

  // --- Build explanation text ---
  function buildExplanation(data, fuzzy) {
    const lines = [];

    if (Number.isNaN(data.age) || data.age < 1 || data.age > 149) {
      return "Please enter a valid age between 1 and 149 to generate a recommendation.";
    }

    lines.push(`Age noted: ${data.age} years.`);

    if (typeof data.stage === "number" && data.stage >= 0 && data.stage <= 4) {
      lines.push(
        `VisionGuard model stage: ${data.stage} (0 = no DR, 4 = most severe).`
      );
    } else {
      lines.push(
        "No numeric stage was provided. The recommendation is based on questionnaire answers only."
      );
    }

    if (data.tested === "yes") {
      lines.push("You reported that you have been tested for diabetes.");
      if (data.lastTestDate) {
        lines.push(`Last test date: ${data.lastTestDate}.`);
      }
      if (data.testReadingRaw) {
        lines.push(`Last reported reading: ${data.testReadingRaw}.`);
      }
    } else {
      lines.push("You reported that you have not been tested for diabetes.");
    }

    if (data.family === "yes") {
      lines.push(
        "You reported a family history of diabetes, which increases your baseline risk."
      );
    } else {
      lines.push(
        "You did not report a family history of diabetes. Risk may still exist due to age, lifestyle, and other factors."
      );
      if (data.lastDoctorVisit) {
        lines.push(
          `Your last general doctor visit was on ${data.lastDoctorVisit}. If it has been a long time, consider a check-up.`
        );
      }
    }

    // Add fuzzy-based follow-up recommendation
    if (fuzzy.label === "Low") {
      lines.push(
        "Overall, the neuro-fuzzy system estimates your short-term eye risk as low. Still, routine eye checks and healthy habits are recommended."
      );
    } else if (fuzzy.label === "Moderate") {
      lines.push(
        "The neuro-fuzzy system estimates a moderate risk. Scheduling a non-urgent appointment with an eye specialist and discussing blood sugar testing may be helpful."
      );
    } else if (fuzzy.label === "High") {
      lines.push(
        "The neuro-fuzzy system estimates a high risk. You should book an eye examination in the near future and speak with a clinician about blood sugar control."
      );
    } else {
      lines.push(
        "The neuro-fuzzy system estimates an urgent level of risk. Please seek prompt evaluation from an eye specialist or healthcare provider, especially if you notice any vision changes."
      );
    }

    lines.push(
      "This tool is for information only and does not replace a professional medical diagnosis."
    );

    return lines.join(" ");
  }

  // --- Render output with styling ---
  function renderOutput(data, fuzzy) {
    output.classList.remove(
      "risk-low",
      "risk-moderate",
      "risk-high",
      "risk-urgent"
    );
    output.classList.add(fuzzy.cssClass);

    const explanation = buildExplanation(data, fuzzy);

    output.innerHTML = `
      <span class="risk-chip">Risk: ${fuzzy.label}</span>
      <span class="score-number">
        (score ≈ ${(fuzzy.risk * 100).toFixed(0)} / 100)
      </span>
      <br /><br />
      ${explanation}
    `;
  }

  // --- Button: Get recommendation ---
  recommendBtn.addEventListener("click", () => {
    const data = getFormData();
    const fuzzy = computeFuzzyRisk(data);
    renderOutput(data, fuzzy);
  });

  // --- Button: Reset form ---
  resetBtn.addEventListener("click", () => {
    form.reset();

    testedFields.style.display = "none";
    testedFields.setAttribute("aria-hidden", "true");
    lastVisitIfNoFamily.style.display = "none";
    lastVisitIfNoFamily.setAttribute("aria-hidden", "true");

    output.classList.remove(
      "risk-low",
      "risk-moderate",
      "risk-high",
      "risk-urgent"
    );
    output.textContent = "Recommendation will appear here.";
  });
});
