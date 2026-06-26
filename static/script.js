// Mapping of education levels to years of education (education_num)
const EDUCATION_NUM_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
};

// Default selections for dropdowns to keep form-filling easy
const SYSTEM_DEFAULTS = {
    "sex": "Male",
    "race": "White",
    "native_country": "United-States",
    "education": "Bachelors",
    "workclass": "Private",
    "occupation": "Exec-managerial",
    "marital_status": "Married-civ-spouse",
    "relationship": "Husband"
};

// UI Elements
const hoursSlider = document.getElementById('hours_per_week');
const hoursDisplay = document.getElementById('hours-display');
const evaluationForm = document.getElementById('evaluation-form');
const formScreen = document.getElementById('form-screen');
const loadingScreen = document.getElementById('loading-screen');
const resultsScreen = document.getElementById('results-screen');
const loadingStatusText = document.querySelector('.loading-status');
const btnReEvaluate = document.getElementById('btn-re-evaluate');

// Result Screen Display Elements
const displayIncome = document.getElementById('display-income');
const displayTierBadge = document.getElementById('display-tier-badge');
const displayTierName = document.getElementById('display-tier-name');
const displayTierDesc = document.getElementById('display-tier-desc');
const productsContainer = document.getElementById('products-container');

// App State
let categoricalCategories = {};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    // 1. Listen to hours range changes
    hoursSlider.addEventListener('input', (e) => {
        hoursDisplay.textContent = e.target.value;
    });

    // 2. Load categorical inputs from the backend
    fetchCategoricalData();

    // 3. Form Submission handler
    evaluationForm.addEventListener('submit', handleFormSubmit);

    // 4. Back button handler
    btnReEvaluate.addEventListener('click', () => {
        switchScreen(formScreen);
    });
});

// Fetch categories from Flask endpoint
async function fetchCategoricalData() {
    try {
        const response = await fetch('/api/categories');
        if (!response.ok) throw new Error("Failed to fetch dropdown categories");
        categoricalCategories = await response.json();
        
        // Populate select elements
        populateDropdown('sex', categoricalCategories.sex);
        populateDropdown('race', categoricalCategories.race);
        populateDropdown('native_country', categoricalCategories.native_country);
        populateDropdown('education', categoricalCategories.education);
        populateDropdown('workclass', categoricalCategories.workclass);
        populateDropdown('occupation', categoricalCategories.occupation);
        populateDropdown('marital_status', categoricalCategories.marital_status);
        populateDropdown('relationship', categoricalCategories.relationship);

    } catch (error) {
        console.error("Error loading dropdown data:", error);
        alert("Could not load category configurations from backend. Please ensure Flask server is running.");
    }
}

// Populate individual select elements and set baseline defaults
function populateDropdown(elementId, optionsList) {
    const selectEl = document.getElementById(elementId);
    if (!selectEl || !optionsList) return;
    
    selectEl.innerHTML = ''; // clear loading state
    
    optionsList.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = formatDisplayName(opt);
        selectEl.appendChild(option);
    });

    // Apply predefined default value if present
    const defaultValue = SYSTEM_DEFAULTS[elementId];
    if (defaultValue && optionsList.includes(defaultValue)) {
        selectEl.value = defaultValue;
    }
}

// Beautify raw census values for dropdown display (e.g. Married-civ-spouse -> Married (Civ-Spouse))
function formatDisplayName(value) {
    if (!value) return '';
    // Format special categories for user-friendly display
    let display = value.replace(/-/g, ' ');
    // Capitalize first letter of each word
    return display.replace(/\b\w/g, c => c.toUpperCase());
}

// Screen switching with clean class triggers
function switchScreen(activeScreen) {
    const allScreens = [formScreen, loadingScreen, resultsScreen];
    allScreens.forEach(screen => {
        screen.classList.remove('active');
        screen.style.display = 'none';
    });
    
    activeScreen.style.display = activeScreen.id === 'loading-screen' ? 'flex' : 'block';
    setTimeout(() => {
        activeScreen.classList.add('active');
    }, 50);
}

// Handle evaluation form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Switch to loading screen
    switchScreen(loadingScreen);
    
    // Simulate loading transitions for high-end feel
    const loadingStatusUpdates = [
        "Coalescing professional metrics...",
        "Querying XGBoost core decision nodes...",
        "Validating income prediction confidence...",
        "Synthesizing customized financial product listings..."
    ];
    
    let statusIndex = 0;
    const statusInterval = setInterval(() => {
        if (statusIndex < loadingStatusUpdates.length) {
            loadingStatusText.textContent = loadingStatusUpdates[statusIndex];
            statusIndex++;
        }
    }, 600);

    // Collect Form Data
    const formData = new FormData(evaluationForm);
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = value;
    });

    // Auto-calculate the education_num mapping based on education selection
    const selectedEducation = payload['education'];
    payload['education_num'] = EDUCATION_NUM_MAP[selectedEducation] || 9; // fallback to HS-grad (9)

    // Generate arbitrary fnlwgt census weight (not used for logic but required by XGBoost structure)
    payload['fnlwgt'] = 189000;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        clearInterval(statusInterval);
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Prediction request failed");
        }
        
        const result = await response.json();
        
        // Wait a slight fraction to let loader feel immersive
        setTimeout(() => {
            renderResults(result);
            switchScreen(resultsScreen);
        }, 300);

    } catch (error) {
        clearInterval(statusInterval);
        console.error("Evaluation Error:", error);
        alert(`Assessment Error: ${error.message}`);
        switchScreen(formScreen);
    }
}

// Render Results on the dashboard screen
function renderResults(result) {
    // Format income value
    const formattedIncome = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
    }).format(result.estimated_income);
    
    displayIncome.textContent = `${formattedIncome} / yr`;
    
    // Update Tier Badge
    displayTierBadge.textContent = result.tier;
    displayTierBadge.className = `tier-badge tier-badge-${result.tier.toLowerCase()}`;
    
    // Update Banner Details
    displayTierName.textContent = result.tier_name;
    
    let descriptionText = "";
    if (result.tier === "Starter") {
        descriptionText = `Based on your profile indices, you qualify for our Starter Banking services. These packages are specifically designed to help build credit, automate emergency funds, and provide financial coaching with zero annual fees.`;
    } else if (result.tier === "Standard") {
        descriptionText = `Your financial indicators place you in our Standard Growth Tier. You are pre-qualified for personal expansion loans, standard rewards credit cards, auto financing options, and low-entry mutual fund allocations.`;
    } else if (result.tier === "Premium") {
        descriptionText = `Congratulations. Your profile meets the criteria for our Premium Advantage Account. You qualify for high-limit Gold cards, competitive fixed rate mortgages, flexi-loans up to $50k, and active balanced portfolios.`;
    } else { // Elite
        descriptionText = `Exclusive Invitation. Your strong professional profile places you in our Elite Private Banking Tier. This unlocks high-limit World Elite credit services, bespoke wealth advisory, private equity options, and liquidity lines up to $5M.`;
    }
    displayTierDesc.textContent = descriptionText;
    
    // Update Products Grid
    productsContainer.innerHTML = '';
    
    result.products.forEach(prod => {
        const card = document.createElement('div');
        card.className = 'product-card';
        
        // Extract rate and limit label names depending on product type
        let spec1Label = "Credit Limit";
        let spec2Label = "Interest Rate";
        if (prod.type === "Savings Account" || prod.type === "Savings & Investments") {
            spec1Label = "Minimum Deposit";
            spec2Label = "Target Yield";
        } else if (prod.type === "Financial Coaching") {
            spec1Label = "Cost";
            spec2Label = "Monthly Fee";
        }
        
        const featuresHtml = prod.features.map(feat => `<li>${feat}</li>`).join('');
        
        card.innerHTML = `
            <div>
                <span class="prod-tag">${prod.type}</span>
                <h3 class="prod-title">${prod.name}</h3>
                
                <div class="prod-meta-box">
                    <div class="meta-item">
                        <span class="meta-label">${spec1Label}</span>
                        <span class="meta-value">${prod.limit}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">${spec2Label}</span>
                        <span class="meta-value">${prod.rate}</span>
                    </div>
                </div>
                
                <ul class="prod-features">
                    ${featuresHtml}
                </ul>
            </div>
        `;
        
        productsContainer.appendChild(card);
    });
}

async function loadMetrics() {
    try {

        // your saved json
        const response =
            await fetch("./metrics/metrics.json");

        if (!response.ok)
            throw new Error("metrics.json not found");

        const data = await response.json();

        renderDashboard(data);

        switchScreen(resultsScreen);

    } catch(err) {
        console.error(err);
    }
}

function renderDashboard(data){

    document.getElementById("test-size").textContent =
        data.test_set_size.toLocaleString();

    renderOverallMetrics(
        data.overall_metrics
    );

    renderFairnessMetrics(
        data.fairness_by_group
    );
}


/* ---------- OVERALL ---------- */

function renderOverallMetrics(metrics){

    const container =
        document.getElementById("overall-metrics");

    container.innerHTML="";

    Object.entries(metrics)
        .forEach(([model, values])=>{

        const card =
            document.createElement("div");

        card.className="metric-card";

        card.innerHTML=`

            <h3>${pretty(model)}</h3>

            ${metricLine(
                "Accuracy",
                values.accuracy
            )}

            ${metricLine(
                "Precision",
                values.precision
            )}

            ${metricLine(
                "Recall",
                values.recall
            )}

            ${metricLine(
                "F1",
                values.f1_score
            )}

            ${metricLine(
                "ROC AUC",
                values.roc_auc,
                false
            )}

        `;

        container.appendChild(card);

    });

}



/* ---------- FAIRNESS ---------- */

function renderFairnessMetrics(models){

    const container =
        document.getElementById(
            "fairness-container"
        );

    container.innerHTML="";

    Object.entries(models)
        .forEach(([model, fairness])=>{

        const wrapper =
            document.createElement("div");

        wrapper.className =
            "fairness-card";

        wrapper.innerHTML=`
            <h2>
                ${pretty(model)}
            </h2>
        `;


        ["sex","race"]
        .forEach(group=>{

            const section =
                document.createElement("div");

            section.className =
                "group-section";

            const meta =
                fairness[group];

            section.innerHTML=`

                <div class="group-header">

                    <h3>
                        ${pretty(group)}
                    </h3>

                    <div class="fairness-summary">

                        <span>
                            DPD:
                            ${meta
                            .demographic_parity_difference
                            .toFixed(3)}
                        </span>

                        <span>
                            EOD:
                            ${meta
                            .equalized_odds_difference
                            .toFixed(3)}
                        </span>

                    </div>

                </div>

                <table>

                    <thead>

                    <tr>

                        <th>Group</th>
                        <th>Accuracy</th>
                        <th>Selection</th>
                        <th>TPR</th>
                        <th>FPR</th>

                    </tr>

                    </thead>

                    <tbody>

                    ${
                        Object.entries(
                            meta.by_group
                        )
                        .map(
                        ([name,row])=>`

                        <tr>

                            <td>${name}</td>

                            <td>
                                ${percent(
                                    row.accuracy
                                )}
                            </td>

                            <td>
                                ${percent(
                                    row.selection_rate
                                )}
                            </td>

                            <td>
                                ${percent(
                                    row.true_positive_rate
                                )}
                            </td>

                            <td>
                                ${percent(
                                    row.false_positive_rate
                                )}
                            </td>

                        </tr>

                        `
                        )
                        .join("")
                    }

                    </tbody>

                </table>

            `;

            wrapper.appendChild(
                section
            );

        });

        container.appendChild(
            wrapper
        );

    });

}



/* ---------- HELPERS ---------- */

function percent(v){

    return (
        v*100
    ).toFixed(1)+"%";

}

function metricLine(
    label,
    value,
    pct=true
){

return `
<div class="metric-row">

<span>${label}</span>

<strong>
${
pct
? percent(value)
: value.toFixed(3)
}
</strong>

</div>
`;

}

function pretty(v){

return v
.replaceAll("_"," ")
.replace(/\b\w/g,
m=>m.toUpperCase());

}


document.addEventListener(
    "DOMContentLoaded",
    loadMetrics
);