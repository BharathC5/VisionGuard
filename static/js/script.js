// Smooth scroll to the corresponding section on click of navigation buttons
document
  .querySelectorAll(".navigation-elements a,.intro-section a")
  .forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = e.currentTarget.getAttribute("href"); // Get the href attribute (section ID)
      const targetSection = document.querySelector(targetId);

      // Get the height of the navbar
      const navbarHeight = document.querySelector(".navigation").offsetHeight;

      // Smooth scroll to the target section
      if (targetSection) {
        window.scrollTo({
          top: targetSection.offsetTop - navbarHeight + 1,
          behavior: "smooth",
        });
      }
    });
  });

// Smooth scroll for buttons in intro-section and navigation
document
  .querySelectorAll(".intro-section button, .navigation button")
  .forEach((button) => {
    button.addEventListener("click", (e) => {
      e.preventDefault();
      const targetLink = e.currentTarget.closest("a");
      if (targetLink) {
        const targetId = targetLink.getAttribute("href");
        const targetSection = document.querySelector(targetId);

        const navbarHeight = document.querySelector(".navigation").offsetHeight;

        // Smooth scroll to the target section
        if (targetSection) {
          window.scrollTo({
            top: targetSection.offsetTop - navbarHeight + 1,
            behavior: "smooth",
          });
        }
      }
    });
  });

// NavBar color change on scroll
window.addEventListener("scroll", () => {
  const navbar = document.querySelector(".navigation");
  const aboutSection = document.querySelector(".about-section");

  if (navbar && aboutSection) {
    const aboutRect = aboutSection.getBoundingClientRect();
    const navRect = navbar.getBoundingClientRect();

    // Check if the .about-section is on screen
    if (navRect.bottom >= aboutRect.top) {
      navbar.style.backgroundColor = "#ffffff";
    } else {
      navbar.style.backgroundColor = "#d6f8d7";
    }
  }
});

// NavBar scroll state change
const scrollThreshold = 100; // Define a threshold for scroll class toggle
const navbar = document.querySelector(".navigation");

window.addEventListener("scroll", () => {
  if (navbar) {
    if (window.scrollY > scrollThreshold) {
      navbar.classList.add("scrolled"); // Add 'scrolled' class
    } else {
      navbar.classList.remove("scrolled"); // Remove 'scrolled' class
    }
  }
});

// Drop area functionality with ML model integration
const dropArea = document.getElementById("drop-area");
if (dropArea) {
  const desc = dropArea.querySelector(".desc"); // Description text
  const resultOutput = document.querySelector(".stage-result-output"); // Result output
  const resultDescription = document.querySelector(".stage-result-description"); // Result description

  // Prevent default drag behaviors to enable drop
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
  }

  // Highlight drop area on drag over
  dropArea.addEventListener("dragover", () => {
    dropArea.classList.add("drag-over");
    if (desc) desc.textContent = "Release to upload retinal image";
  });

  // Remove highlight on drag leave
  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("drag-over");
    if (desc) desc.textContent = "Click or Drag & Drop Retinal Image";
  });

  // Handle file drop
  dropArea.addEventListener("drop", (event) => {
    dropArea.classList.remove("drag-over");
    const files = event.dataTransfer.files;
    handleFiles(files);
  });

  // Handle click to open file selector
  dropArea.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*"; // Only allow image files
    fileInput.multiple = false;
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    fileInput.addEventListener("change", (event) => {
      const files = event.target.files;
      handleFiles(files);
      fileInput.remove(); // Cleanup
    });

    fileInput.click();
  });

  // Function to predict diabetic retinopathy stage
  async function predictRetinopathyStage(file) {
    // Show loading state
    if (resultOutput) resultOutput.textContent = "Analyzing...";
    if (resultDescription)
      resultDescription.textContent = "Processing retinal image";

    try {
      // Convert file to base64 or prepare for model input
      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = async (e) => {
        const imageBase64 = e.target.result;

        try {
          console.log("Sending image for prediction:", {
            fileType: file.type,
            fileSize: file.size,
          });

          const response = await fetch("http://localhost:5000/home", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              image: imageBase64,
            }),
          });

          console.log("Response status:", response.status);

          if (!response.ok) {
            const errorText = await response.text();
            console.error("Error response:", errorText);
            throw new Error("Prediction failed: " + errorText);
          }

          const predictionResult = await response.json();
          console.log("Prediction result:", predictionResult);

          // Update UI with prediction results
          if (resultOutput)
            resultOutput.textContent = `Stage: ${predictionResult.stage}`;

          // Provide more detailed description based on stage
          const stageDescriptions = {
            0: "No Diabetic Retinopathy Detected",
            1: "Mild Non-Proliferative Diabetic Retinopathy",
            2: "Moderate Non-Proliferative Diabetic Retinopathy",
            3: "Severe Non-Proliferative Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy",
          };

          if (resultDescription) {
            resultDescription.textContent =
              stageDescriptions[predictionResult.stage] ||
              "Unable to classify diabetic retinopathy stage";
          }

          // Optional: Display original image
          displayUploadedImage(imageBase64);
        } catch (predictionError) {
          console.error("Detailed Prediction Error:", predictionError);
          if (resultOutput) resultOutput.textContent = "Prediction Failed";
          if (resultDescription)
            resultDescription.textContent =
              predictionError.message || "Unable to process the image";
        }
      };
    } catch (error) {
      console.error("Initial Error:", error);
      if (resultOutput) resultOutput.textContent = "Error Processing Image";
      if (resultDescription)
        resultDescription.textContent =
          "Please try again with a different image";
    }
  }

  // Function to display uploaded image
  function displayUploadedImage(imageBase64) {
    // Create an image element to display the uploaded image
    const imageContainer = document.querySelector(".uploaded-image-container");
    if (imageContainer) {
      imageContainer.innerHTML = ""; // Clear previous image
      const imgElement = document.createElement("img");
      imgElement.src = imageBase64;
      imgElement.alt = "Uploaded Retinal Image";
      imgElement.classList.add("uploaded-image");
      imageContainer.appendChild(imgElement);
    }
  }

  // Centralized function to handle file processing
  function handleFiles(files) {
    if (!files || files.length === 0) {
      resetUI("No files selected");
      return;
    }

    // Validate file selection
    const file = files[0];
    const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
    const maxFileSize = 5 * 1024 * 1024; // 5MB max file size

    if (!allowedTypes.includes(file.type)) {
      resetUI("Invalid file type. Please upload JPEG or PNG images.");
      return;
    }

    if (file.size > maxFileSize) {
      resetUI("File is too large. Maximum size is 5MB.");
      return;
    }

    // Update UI with file details and start prediction
    if (desc) desc.textContent = `Uploaded: ${file.name}`;

    // Call prediction function
    predictRetinopathyStage(file);
  }

  // Centralized UI reset function
  function resetUI(message) {
    if (desc)
      desc.textContent = message || "Click or Drag & Drop Retinal Image";
    if (resultOutput) resultOutput.textContent = "No Files";
    if (resultDescription)
      resultDescription.textContent = "Please upload a valid retinal image.";
  }
}

function displayUploadedImage(imageBase64) {
  // Create an image element to display the uploaded image
  const imageContainer = document.querySelector(".uploaded-image-container");
  if (imageContainer) {
    imageContainer.innerHTML = ""; // Clear any previous images
    const imgElement = document.createElement("img");
    imgElement.src = imageBase64;
    imgElement.alt = "Uploaded Retinal Image";
    imgElement.classList.add("uploaded-image");
    imageContainer.appendChild(imgElement);
  } else {
    console.error("Image container element not found.");
  }
}

// ===== quick form integration script =====
(function () {
  // Elements
  const ageChips = document.querySelectorAll("#patient-quick-form .age-chip");
  const ageHidden = document.getElementById("quick_age_group");
  const checkedRadios = document.querySelectorAll(
    'input[name="quick_checked"]'
  );
  const checkedSub = document.getElementById("quick_checked_sub");
  const pickBtn = document.getElementById("quick_pick_file");
  const clearBtn = document.getElementById("quick_clear");
  const statusEl = document.getElementById("quick_status");
  const resultLabel = document.getElementById("quick_result_label");
  const resultDesc = document.getElementById("quick_result_desc");
  const uploadedImageContainer = document.querySelector(
    "#patient-quick-form .uploaded-image-container"
  );

  // age chips behaviour
  ageChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      ageChips.forEach((c) => {
        c.style.background = "#fff";
        c.setAttribute("aria-pressed", "false");
        c.style.border = "1px solid #e6eefc";
      });
      chip.style.background = "#eef6ff";
      chip.style.border = "1px solid #2b6cff";
      chip.setAttribute("aria-pressed", "true");
      ageHidden.value = chip.dataset.value;
    });
  });

  // show/hide checked-subfields
  function updateChecked() {
    const yes = document.querySelector(
      'input[name="quick_checked"][value="yes"]'
    );
    const isYes = yes && yes.checked;
    checkedSub.style.display = isYes ? "block" : "none";
  }
  checkedRadios.forEach((r) => r.addEventListener("change", updateChecked));
  updateChecked();

  // helper to collect meta
  function collectQuickMeta() {
    return {
      ageGroup: document.getElementById("quick_age_group")?.value || "",
      checkedDiabetes:
        document.querySelector('input[name="quick_checked"]:checked')?.value ===
        "yes",
      checkedWhenMonths:
        Number(document.getElementById("quick_checked_when")?.value || 0) ||
        null,
      checkedReadingMgDl:
        Number(document.getElementById("quick_checked_reading")?.value || 0) ||
        null,
      family: document.getElementById("quick_family")?.value || "none",
    };
  }

  // open file picker and send file to wrapped predict function
  pickBtn.addEventListener("click", () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/jpeg,image/png";
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);
    fileInput.addEventListener("change", (e) => {
      const files = e.target.files;
      if (files && files.length) {
        const file = files[0];
        statusEl.textContent = "Uploading...";
        // wrap predictRetinopathyStage so it receives meta as second argument (if function supports it)
        if (typeof predictRetinopathyStage === "function") {
          // If original function accepts (file, meta) we pass meta; otherwise we fallback to building our own fetch call
          try {
            // try calling with two args; many server wrappers ignore extra args, but in case your implementation accepts meta...
            predictRetinopathyStage(file, collectQuickMeta());
            statusEl.textContent = `Sent for prediction: ${file.name}`;
          } catch (err) {
            // fallback: build base64 and POST ourselves (mirrors earlier predict function)
            fallbackSend(file);
          }
        } else {
          fallbackSend(file);
        }
      }
      fileInput.remove();
    });
    fileInput.click();
  });

  clearBtn.addEventListener("click", () => {
    // reset UI
    statusEl.textContent = "Cleared";
    resultLabel.textContent = "";
    resultDesc.textContent = "";
    if (uploadedImageContainer) uploadedImageContainer.innerHTML = "";
    // reset form to defaults
    document.querySelector(
      'input[name="quick_checked"][value="no"]'
    ).checked = true;
    updateChecked();
    // age chip reset to first
    if (ageChips[0]) ageChips[0].click();
    document.getElementById("quick_family").value = "none";
  });

  // fallbackSend: reads file base64 and POSTs payload (if predictRetinopathyStage isn't available)
  async function fallbackSend(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async (e) => {
      const imageBase64 = e.target.result;
      // display
      if (uploadedImageContainer) {
        uploadedImageContainer.innerHTML = "";
        const img = document.createElement("img");
        img.src = imageBase64;
        img.alt = "Uploaded";
        img.style.maxWidth = "220px";
        img.style.borderRadius = "8px";
        img.style.border = "1px solid #eef4ff";
        uploadedImageContainer.appendChild(img);
      }
      const payload = { image: imageBase64, meta: collectQuickMeta() };
      try {
        const resp = await fetch("http://localhost:5000/home", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error("Server error");
        const res = await resp.json();
        resultLabel.textContent = `Stage: ${res.stage ?? "N/A"}`;
        const map = {
          0: "No DR",
          1: "Mild",
          2: "Moderate",
          3: "Severe",
          4: "Proliferative",
        };
        resultDesc.textContent = map[res.stage] || res.message || "";
        statusEl.textContent = `Prediction received`;
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Prediction failed";
      }
    };
    reader.onerror = () => {
      statusEl.textContent = "Failed to read file";
    };
  }
})();

/* Form dynamic UI and recommendation logic.
   Include this in your static/js/script.js or a new file and ensure it's loaded after the form HTML. */

(function () {
  // Helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const formatDate = (d) => d.toLocaleDateString();

  // Elements
  const testedRadios = $$('input[name="tested"]');
  const familyRadios = $$('input[name="familyHistory"]');
  const testedYesFields = $("#tested-yes-fields");
  const lastVisitIfNoFamily = $("#last-visit-if-no-family");
  const computeBtn = $("#computeRecommendationBtn");
  const output = $("#recommendationOutput");
  const form = $("#visionForm");

  // Show/hide helper
  function setVisible(el, visible) {
    if (!el) return;
    el.style.display = visible ? "block" : "none";
    el.setAttribute("aria-hidden", visible ? "false" : "true");
  }

  // Attach listeners to tested radios
  testedRadios.forEach((r) =>
    r.addEventListener("change", () => {
      setVisible(testedYesFields, r.value === "yes");
    })
  );

  // Attach listeners to family history radios
  familyRadios.forEach((r) =>
    r.addEventListener("change", () => {
      // show the "last doctor visit" only when family history is No
      setVisible(lastVisitIfNoFamily, r.value === "no");
    })
  );

  // Initialize visibility
  setVisible(
    testedYesFields,
    $('input[name="tested"]:checked').value === "yes"
  );
  setVisible(
    lastVisitIfNoFamily,
    $('input[name="familyHistory"]:checked').value === "no"
  );

  // Validate form inputs (basic)
  function validateForm() {
    const age = parseInt($("#age").value, 10);
    if (!age || age <= 0 || age >= 150) {
      return { ok: false, message: "Please enter a valid age (1–149)." };
    }
    // If tested yes, optional date/reading, but if they enter reading ensure numeric
    if ($('input[name="tested"]:checked').value === "yes") {
      const readingVal = $("#testReading").value;
      if (readingVal && isNaN(Number(readingVal))) {
        return { ok: false, message: "Test reading must be numeric." };
      }
    }
    return { ok: true };
  }

  // Core recommendation logic
  // stage : integer 0..4
  // returns object { message, suggestedDate (Date), severityBucket }
  function computeRecommendation(stage, formValues) {
    // Determine some flags from form
    const hasTested = formValues.tested === "yes";
    let reading = null;
    if (hasTested && formValues.testReading)
      reading = Number(formValues.testReading);
    const familyHistory = formValues.familyHistory === "yes";
    const lastTestDate = formValues.lastTestDate
      ? new Date(formValues.lastTestDate)
      : null;
    const lastDoctorVisit = formValues.lastDoctorVisit
      ? new Date(formValues.lastDoctorVisit)
      : null;

    // Utility to add days
    const addDays = (d, days) => {
      const r = new Date(d.valueOf());
      r.setDate(r.getDate() + days);
      return r;
    };

    const today = new Date();

    // Decide window (days) default
    let daysToVisit;
    let severity;

    // Urgent stages
    if (stage >= 4) {
      daysToVisit = 1; // immediate/within 24-72 hours
      severity = "urgent";
    } else if (stage === 3) {
      daysToVisit = 14; // 1-2 weeks
      severity = "high";
    } else if (stage === 2) {
      daysToVisit = 30;
      severity = "moderate";
    } else if (stage === 1) {
      daysToVisit = 90;
      severity = "mild";
    } else {
      // stage 0
      // If patient has diabetes (use reading heuristic) or family history -> sooner
      let likelyDiabetes = false;
      // Simple numeric heuristic (NOT clinical diagnosis): fasting >=126 or random >=200 (if user didn't specify, we still use simple threshold)
      // Because we don't know units/fasting vs random, we'll use a conservative simple rule:
      if (reading !== null) {
        if (reading >= 200) likelyDiabetes = true; // strong sign
        else if (reading >= 126) likelyDiabetes = true; // possible
      }
      if (hasTested && lastTestDate) {
        // if last test is more than 1 year old, recommend sooner
        const msInYear = 365 * 24 * 3600 * 1000;
        if (today - lastTestDate > msInYear) {
          // nudge sooner
          daysToVisit = 90; // 3 months
          severity = "mild";
        }
      }
      // Default for stage 0 if not set above
      if (!daysToVisit) {
        if (likelyDiabetes || familyHistory) {
          daysToVisit = 180; // 6 months
          severity = "low";
        } else {
          daysToVisit = 365; // annual routine
          severity = "routine";
        }
      }
    }

    // Further nudges: if user hasn't seen doctor in long time, recommend sooner
    if (lastDoctorVisit) {
      const msInYear = 365 * 24 * 3600 * 1000;
      const today = new Date();
      if (today - lastDoctorVisit > msInYear) {
        // bring date forward by 30 days if recommendation was routine/annual
        if (daysToVisit >= 365) daysToVisit = 90;
      }
    }

    const suggestedDate = addDays(today, Math.max(1, Math.round(daysToVisit)));
    const message = buildMessage(stage, severity, suggestedDate, daysToVisit, {
      likelyDiabetes,
      reading,
      lastTestDate,
      familyHistory,
    });

    return { message, suggestedDate, severity, daysToVisit };
  }

  function buildMessage(stage, severity, suggestedDate, daysToVisit, extras) {
    const parts = [];

    // Basic stage-based text
    if (stage >= 4) {
      parts.push(
        "Diagnosis severity: **Proliferative / advanced** — immediate attention recommended."
      );
      parts.push(
        "Please contact an ophthalmologist or emergency eye service as soon as possible (within 24–72 hours)."
      );
    } else if (stage === 3) {
      parts.push(
        "Diagnosis severity: **Severe non-proliferative**. Specialist appointment recommended quickly."
      );
    } else if (stage === 2) {
      parts.push(
        "Diagnosis severity: **Moderate**. Schedule an ophthalmology visit within about 1 month."
      );
    } else if (stage === 1) {
      parts.push(
        "Diagnosis severity: **Mild**. Follow-up with your eye doctor within about 3 months is recommended."
      );
    } else {
      parts.push(
        "Diagnosis severity: **No diabetic retinopathy detected (Stage 0)**."
      );
    }

    // Add contextual nudges
    if (extras.likelyDiabetes) {
      parts.push(
        "Your blood-sugar reading suggests possible elevated glucose. This is NOT a formal diagnosis from this tool — please see a physician for confirmation."
      );
    } else if (extras.reading !== null) {
      parts.push(
        "You provided a numeric test reading. If uncertain about units or fasting/random status, mention this to your clinician."
      );
    }

    if (extras.lastTestDate) {
      parts.push(
        `Last diabetes test: ${formatDate(new Date(extras.lastTestDate))}.`
      );
    }

    if (extras.familyHistory) {
      parts.push(
        "Family history of diabetes: yes — this increases your risk and suggests closer monitoring."
      );
    }

    parts.push(
      `Recommended next appointment: **by ${formatDate(
        suggestedDate
      )}** (in about ${Math.round(daysToVisit)} day(s)).`
    );

    parts.push(
      "⚠️ This recommendation is informational only and not a substitute for professional medical advice."
    );

    return parts.join("\n\n");
  }

  // Called when compute button clicked OR when backend diagnosis is available.
  // If stage is not provided, uses stage = 0 as default (you should call onDiagnosisResult when you have actual stage).
  function processAndShow(stage = null) {
    const v = validateForm();
    if (!v.ok) {
      output.textContent = v.message;
      output.style.color = "crimson";
      alert(v.message);
      return;
    }

    // Gather form values
    const formValues = {
      age: Number($("#age").value),
      tested: $('input[name="tested"]:checked').value,
      lastTestDate: $("#lastTestDate").value || null,
      testReading: $("#testReading").value || null,
      familyHistory: $('input[name="familyHistory"]:checked').value,
      lastDoctorVisit: $("#lastDoctorVisit").value || null,
    };

    // If caller didn't pass stage, try to read existing stage displayed on page (.stage-result-output or similar)
    let effectiveStage = null;
    if (stage === null) {
      // Try to parse from DOM (if your backend placed stage into .stage-result-output or similar)
      const stageTextEl = document.querySelector(".stage-result-output");
      if (stageTextEl) {
        const txt = stageTextEl.textContent || stageTextEl.innerText || "";
        // try extract 0-4
        const m = txt.match(/\b([0-4])\b/);
        if (m) effectiveStage = Number(m[1]);
      }
    } else {
      effectiveStage = Number(stage);
    }
    // fallback
    if (effectiveStage === null || isNaN(effectiveStage)) effectiveStage = 0;

    const recom = computeRecommendation(effectiveStage, formValues);

    // Show friendly output and a browser alert
    output.style.color = "#0f172a";
    output.innerHTML = recom.message
      .replace(/\n\n/g, "<br><br>")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Also show an alert summarizing the next visit date
    alert(
      `Recommendation: ${recom.severity.toUpperCase()}\nNext suggested appointment by: ${formatDate(
        recom.suggestedDate
      )}\n\nSee the message box for details.`
    );

    // Optionally, you can also scroll to output
    output.scrollIntoView({ behavior: "smooth", block: "center" });

    return recom;
  }

  // Attach compute button
  computeBtn.addEventListener("click", () => processAndShow(null));

  // Expose a global handler that your existing backend callback can call
  // Example usage in your existing code after getting JSON { stage: X } from Flask:
  //    window.onDiagnosisResult(response.stage);
  window.onDiagnosisResult = function (stageFromBackend) {
    try {
      const stage = Number(stageFromBackend);
      if (isNaN(stage) || stage < 0 || stage > 4) {
        console.warn(
          "onDiagnosisResult called with invalid stage:",
          stageFromBackend
        );
      }
      // compute and show recommendation using the form and the provided stage
      processAndShow(stage);
    } catch (err) {
      console.error("Error in onDiagnosisResult:", err);
    }
  };

  // Optional: if your backend triggers a custom event, you can listen to it here:
  // document.addEventListener('visionGuardDiagnosis', e => window.onDiagnosisResult(e.detail.stage));
})();

(function () {
  // Simple helpers
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const form = $("#vgForm");
  const testedRadios = $$('input[name="tested"]');
  const familyRadios = $$('input[name="family"]');
  const testedFields = $("#testedFields");
  const lastVisitIfNoFamily = $("#lastVisitIfNoFamily");
  const output = $("#output");

  // Show/hide helper
  function show(el, yes) {
    if (!el) return;
    el.style.display = yes ? "block" : "none";
    el.setAttribute("aria-hidden", yes ? "false" : "true");
  }

  // init visibility
  show(testedFields, $('input[name="tested"]:checked').value === "yes");
  show(lastVisitIfNoFamily, $('input[name="family"]:checked').value === "no");

  // listeners
  testedRadios.forEach((r) =>
    r.addEventListener("change", (e) => {
      show(testedFields, e.target.value === "yes");
    })
  );
  familyRadios.forEach((r) =>
    r.addEventListener("change", (e) => {
      show(lastVisitIfNoFamily, e.target.value === "no");
    })
  );

  // Basic validation
  function validate() {
    const age = Number($("#age").value);
    if (!age || isNaN(age) || age < 1 || age > 149) {
      return {
        ok: false,
        msg: "Please enter a valid age between 1 and 149.",
      };
    }
    const tested = $('input[name="tested"]:checked').value === "yes";
    if (tested) {
      const reading = $("#testReading").value.trim();
      if (reading && isNaN(Number(reading.replace(/[^0-9.\-]/g, "")))) {
        return {
          ok: false,
          msg: "Test reading must be numeric (you can include units but keep numeric part first).",
        };
      }
    }
    return { ok: true };
  }

  // Basic recommendation logic (same heuristics as earlier)
  function computeRecommendation(stage, values) {
    // stage expected 0..4
    stage = Number(stage);
    if (isNaN(stage) || stage < 0 || stage > 4) stage = 0;

    const today = new Date();
    const parseDate = (d) => (d ? new Date(d) : null);
    const lastTest = parseDate(values.lastTestDate);
    const lastVisit = parseDate(values.lastDoctorVisit);
    const readingStr = (values.testReading || "").trim();
    const reading = readingStr
      ? Number(readingStr.replace(/[^0-9.\-]/g, ""))
      : null;
    const family = values.family === "yes";
    let days = 365;
    let severity = "routine";

    if (stage >= 4) {
      days = 1;
      severity = "urgent";
    } else if (stage === 3) {
      days = 14;
      severity = "high";
    } else if (stage === 2) {
      days = 30;
      severity = "moderate";
    } else if (stage === 1) {
      days = 90;
      severity = "mild";
    } else {
      // stage 0
      // reading heuristic
      let likelyDiabetes = false;
      if (reading !== null && !isNaN(reading)) {
        if (reading >= 200) likelyDiabetes = true;
        else if (reading >= 126) likelyDiabetes = true;
      }
      if (likelyDiabetes || family) {
        days = 180;
        severity = "low";
      } else days = 365;
      // if lastTest older than a year, bring forward:
      if (lastTest && today - lastTest > 365 * 24 * 3600 * 1000)
        days = Math.min(days, 90);
    }

    // if last doctor visit > 1 year, nudge sooner
    if (lastVisit && today - lastVisit > 365 * 24 * 3600 * 1000) {
      if (days > 90) days = 90;
    }

    const suggested = new Date();
    suggested.setDate(suggested.getDate() + Math.max(1, Math.round(days)));

    // build message
    const parts = [];
    const stageMsg =
      [
        "No DR detected (stage 0)",
        "Mild (stage 1)",
        "Moderate (stage 2)",
        "Severe (stage 3)",
        "Proliferative (stage 4)",
      ][stage] || `Stage ${stage}`;
    parts.push(`Diagnosis: ${stageMsg}`);
    if (stage >= 4)
      parts.push(
        "This appears urgent — seek ophthalmology care immediately (within 24–72 hours)."
      );
    else
      parts.push(
        `Suggested follow-up: by ${suggested.toDateString()} (in ~${Math.round(
          days
        )} day(s)).`
      );
    if (reading !== null)
      parts.push(
        `Provided glucose reading: ${reading}${
          readingStr.replace(/[0-9.\-]/g, "") || ""
        }`
      );
    if (family) parts.push("Family history: yes — consider closer monitoring.");
    parts.push(
      "⚠️ This is informational only — consult a clinician for medical advice."
    );

    return {
      text: parts.join("\n\n"),
      suggestedDate: suggested,
      severity,
      days,
    };
  }

  // Process and show; used by button and by global handler below
  function processAndShow(stageFromBackend) {
    const v = validate();
    if (!v.ok) {
      alert(v.msg);
      output.textContent = v.msg;
      output.style.color = "crimson";
      return;
    }
    // gather values
    const values = {
      age: $("#age").value,
      tested: $('input[name="tested"]:checked').value,
      lastTestDate: $("#lastTestDate").value || null,
      testReading: $("#testReading").value || null,
      family: $('input[name="family"]:checked').value,
      lastDoctorVisit: $("#lastDoctorVisit").value || null,
    };

    // default stage 0 if backend didn't provide stage
    const stage =
      typeof stageFromBackend !== "undefined" && stageFromBackend !== null
        ? Number(stageFromBackend)
        : 0;
    const res = computeRecommendation(stage, values);

    // show in output and alert summary
    output.innerHTML = res.text.replace(/\n\n/g, "<br><br>");
    output.style.color = "#0f172a";
    alert(
      `Recommendation (${res.severity.toUpperCase()}): next suggested appointment by ${res.suggestedDate.toDateString()}`
    );
    return res;
  }

  // Button
  $("#recommendBtn").addEventListener("click", () => processAndShow(null));
  $("#resetBtn").addEventListener("click", () => {
    form.reset();
    show(testedFields, false);
    show(lastVisitIfNoFamily, true); // default family=no so show lastVisit
    output.textContent = "Recommendation will appear here.";
    output.style.color = "#0f172a";
  });

  // Expose global handler: call this when backend returns JSON { stage: X }
  // Example: after your fetch to Flask returns resp, call window.onDiagnosisResult(resp.stage);
  window.onDiagnosisResult = function (stage) {
    try {
      processAndShow(stage);
    } catch (e) {
      console.error("onDiagnosisResult error:", e);
    }
  };

  // initial state - ensure testedFields & lastVisit visibility match current inputs
  show(testedFields, $('input[name="tested"]:checked').value === "yes");
  show(lastVisitIfNoFamily, $('input[name="family"]:checked').value === "no");
})();
