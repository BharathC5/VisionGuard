// -------------------------
// VisionGuard Form Logic
// -------------------------

(function () {
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));

  const form = $('#vgForm');
  const testedFields = $('#testedFields');
  const lastVisitIfNoFamily = $('#lastVisitIfNoFamily');
  const output = $('#output');

  // Show / hide helper
  function show(el, yes) {
    el.style.display = yes ? 'block' : 'none';
    el.setAttribute('aria-hidden', yes ? 'false' : 'true');
  }

  // Visibility listeners
  $$('input[name="tested"]').forEach(r =>
    r.addEventListener('change', e => {
      show(testedFields, e.target.value === "yes");
    })
  );

  $$('input[name="family"]').forEach(r =>
    r.addEventListener('change', e => {
      show(lastVisitIfNoFamily, e.target.value === "no");
    })
  );

  // Initialize on load
  show(testedFields, $('input[name="tested"]:checked').value === "yes");
  show(lastVisitIfNoFamily, $('input[name="family"]:checked').value === "no");

  // Validation
  function validate() {
    const age = Number($('#age').value);
    if (!age || age < 1 || age > 149) {
      return { ok:false, msg:"Age must be between 1 and 149." };
    }
    const tested = $('input[name="tested"]:checked').value === "yes";
    if (tested) {
      const reading = $('#testReading').value.trim();
      if (reading && isNaN(Number(reading.replace(/[^0-9.\-]/g,'')))) {
        return { ok:false, msg:"Test reading must be numeric." };
      }
    }
    return { ok:true };
  }

  // Compute recommendation
  function compute(stage, values) {
    stage = Number(stage);
    if (isNaN(stage) || stage < 0 || stage > 4) stage = 0;

    const today = new Date();
    let days = 365;
    let severity = "routine";

    // Reading value
    const readingStr = values.testReading.trim();
    const reading = readingStr ? Number(readingStr.replace(/[^0-9.\-]/g,'')) : null;
    const family = values.family === "yes";

    const parseDate = d => d ? new Date(d) : null;
    const lastTest = parseDate(values.lastTestDate);
    const lastVisit = parseDate(values.lastDoctorVisit);

    // Stage logic
    if (stage >= 4)      { days = 1; severity = "urgent"; }
    else if (stage === 3){ days = 14; severity = "high"; }
    else if (stage === 2){ days = 30; severity = "moderate"; }
    else if (stage === 1){ days = 90; severity = "mild"; }
    else {
      let likely = false;
      if (reading !== null) {
        if (reading >= 200) likely = true;
        else if (reading >= 126) likely = true;
      }

      days = likely || family ? 180 : 365;

      if (lastTest && (today - lastTest) > (365 * 24 * 3600 * 1000))
        days = Math.min(days, 90);
    }

    if (lastVisit && (today - lastVisit) > (365 * 24 * 3600 * 1000)) {
      if (days > 90) days = 90;
    }

    const visit = new Date();
    visit.setDate(visit.getDate() + days);

    const lines = [];
    const names = [
      "No DR detected (Stage 0)",
      "Mild (Stage 1)",
      "Moderate (Stage 2)",
      "Severe (Stage 3)",
      "Proliferative (Stage 4)"
    ];

    lines.push("Diagnosis: " + (names[stage] || "Stage " + stage));

    if (stage >= 4) {
      lines.push("URGENT — Visit an ophthalmologist within 24–72 hours.");
    }

    lines.push(`Next suggested check: ${visit.toDateString()} (in ~${days} days)`);

    if (reading !== null)
      lines.push("Your glucose reading: " + readingStr);

    if (family)
      lines.push("Family history: YES — Higher risk.");

    lines.push("Note: This is informational only, not medical advice.");

    return lines.join("\n\n");
  }

  // Process and show
  function run(stageValue) {
    const v = validate();
    if (!v.ok) {
      alert(v.msg);
      output.textContent = v.msg;
      output.style.color = "crimson";
      return;
    }

    const values = {
      age: $('#age').value,
      tested: $('input[name="tested"]:checked').value,
      lastTestDate: $('#lastTestDate').value || "",
      testReading: $('#testReading').value || "",
      family: $('input[name="family"]:checked').value,
      lastDoctorVisit: $('#lastDoctorVisit').value || ""
    };

    const resultText = compute(stageValue, values);
    output.innerHTML = resultText.replace(/\n\n/g,"<br><br>");
    output.style.color = "#0f172a";

    alert("Recommendation ready — see the box below.");
  }

  // Button listener
  $('#recommendBtn').addEventListener('click', () => run(null));

  // Reset button
  $('#resetBtn').addEventListener('click', () => {
    form.reset();
    show(testedFields, false);
    show(lastVisitIfNoFamily, true);
    output.textContent = "Recommendation will appear here.";
  });

  // Global for backend to call
  window.onDiagnosisResult = function(stage) {
    run(stage);
  };

})();
