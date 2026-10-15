import { useEffect, useState } from "react";
import "./App.css";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  LabelList,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";


// ============================================================
// API CONFIGURATION
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// CHART COLORS
// ============================================================

const COLORS = [
  "#756cff",
  "#22b8d6",
  "#9a72e8",
  "#32b982",
];


// ============================================================
// APP
// ============================================================

function App() {

  const [file, setFile] = useState(null);

  const [preview, setPreview] = useState("");

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  // ==========================================================
  // THEME
  // ==========================================================

  const [theme, setTheme] = useState(() => {

    return (
      localStorage.getItem("alzora-theme") ||
      "light"
    );

  });


  useEffect(() => {

    localStorage.setItem(
      "alzora-theme",
      theme
    );

  }, [theme]);


  const toggleTheme = () => {

    setTheme((current) =>
      current === "light"
        ? "dark"
        : "light"
    );

  };


  const isDark = theme === "dark";


  // ==========================================================
  // CLASS NAME CLEANER
  // ==========================================================

  const cleanClass = (name = "") =>
    name.replace(
      "Very mild",
      "Very Mild"
    );


  // ==========================================================
  // FILE HANDLING
  // ==========================================================

  const handleFile = (e) => {

    const selected =
      e.target.files?.[0];

    if (!selected) return;


    // --------------------------------------------------------
    // Validate image
    // --------------------------------------------------------

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/jpg",
      "image/bmp",
      "image/webp",
    ];


    if (!allowedTypes.includes(selected.type)) {

      setError(
        "Please upload a valid MRI image (JPG, JPEG, PNG, BMP or WEBP)."
      );

      return;

    }


    setFile(selected);

    setPreview(
      URL.createObjectURL(selected)
    );

    setResult(null);

    setError("");

  };


  // ==========================================================
  // MRI ANALYSIS
  // ==========================================================

  const analyzeMRI = async () => {

    if (!file) {

      setError(
        "Please upload an MRI image first."
      );

      return;

    }


    setLoading(true);

    setError("");

    setResult(null);


    try {

      // ------------------------------------------------------
      // Create multipart form
      // ------------------------------------------------------

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );


      // ------------------------------------------------------
      // Send MRI to FastAPI
      // ------------------------------------------------------

      const response =
        await fetch(
          `${API_URL}/predict`,
          {
            method: "POST",
            body: formData,
          }
        );


      // ------------------------------------------------------
      // Parse response
      // ------------------------------------------------------

      let data;


      try {

        data =
          await response.json();

      } catch {

        throw new Error(
          `Backend returned an invalid response (HTTP ${response.status}).`
        );

      }


      // ------------------------------------------------------
      // HTTP error
      // ------------------------------------------------------

      if (!response.ok) {

        throw new Error(
          data.detail ||
          `Prediction failed (HTTP ${response.status}).`
        );

      }


      // ------------------------------------------------------
      // Application-level error
      // ------------------------------------------------------

      if (!data.success) {

        throw new Error(
          data.detail ||
          "MRI analysis was not successful."
        );

      }


      // ------------------------------------------------------
      // Save result
      // ------------------------------------------------------

      setResult(data);


      // ------------------------------------------------------
      // Scroll to results
      // ------------------------------------------------------

      setTimeout(() => {

        document
          .getElementById("results")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });

      }, 200);

    } catch (err) {

      console.error(
        "MRI analysis error:",
        err
      );


      // ------------------------------------------------------
      // Connection error
      // ------------------------------------------------------

      if (
        err instanceof TypeError
      ) {

        setError(
          "Unable to connect to Alzora AI. Please make sure the backend is running on port 8000."
        );

      } else {

        setError(
          err.message ||
          "Unable to analyze the MRI image."
        );

      }

    } finally {

      setLoading(false);

    }

  };


  // ==========================================================
  // RESET
  // ==========================================================

  const reset = () => {

    setFile(null);

    setPreview("");

    setResult(null);

    setError("");

  };


  // ==========================================================
  // PROBABILITIES
  // ==========================================================

  const probabilities =
    result?.probabilities
      ? Object.entries(
          result.probabilities
        ).map(
          (
            [name, value],
            index
          ) => ({

            name:
              cleanClass(name),

            value:
              Number(value),

            color:
              COLORS[
                index %
                COLORS.length
              ],

          })
        )
      : [];


  // ==========================================================
  // RANKED PROBABILITIES
  // ==========================================================

  const ranked =
    [...probabilities].sort(
      (a, b) =>
        b.value - a.value
    );


  // ==========================================================
  // PREDICTION
  // ==========================================================

  const prediction =
    cleanClass(
      result?.prediction
    );


  const predictionProbability =
    probabilities.find(
      (item) =>
        item.name ===
        prediction
    )?.value ||
    Number(
      result?.confidence || 0
    );


  // ==========================================================
  // NON-DEMENTED PROBABILITY
  // ==========================================================

  const nonDementedProbability =
    probabilities.find(
      (item) =>
        item.name ===
        "Non Demented"
    )?.value || 0;


  const difference =
    predictionProbability -
    nonDementedProbability;


  const ratio =
    nonDementedProbability > 0
      ? predictionProbability /
        nonDementedProbability
      : null;


  // Prevent unused-variable issues while preserving logic.
  void difference;
  void ratio;


  // ==========================================================
  // DETECTION
  // ==========================================================

  const detection =
    result?.detection ||
    (
      prediction ===
      "Non Demented"
        ? "Not Detected"
        : "Detected"
    );


  const isDetected =
    detection.toLowerCase() ===
    "detected";


  const detectionLabel =
    result?.detection_label ||
    (
      isDetected
        ? "Dementia Indication Detected"
        : "Dementia Indication Not Detected"
    );


  const dementiaProbability =
    result?.dementia_probability !==
    undefined
      ? Number(
          result.dementia_probability
        )
      : Math.max(
          0,
          100 -
            nonDementedProbability
        );


  // ==========================================================
  // BLOCKCHAIN DATA
  // ==========================================================

  const blockchain =
    result?.blockchain || null;


  const blockchainConfirmed =
    blockchain?.status ===
    "confirmed";


  // ==========================================================
  // THEME-AWARE CHART COLORS
  // ==========================================================

  const chart = {

    grid:
      isDark
        ? "#34415e"
        : "#dce3ef",

    axis:
      isDark
        ? "#53617b"
        : "#b8c3d4",

    text:
      isDark
        ? "#aab5cb"
        : "#65738a",

    tooltipBg:
      isDark
        ? "#202b49"
        : "#ffffff",

    tooltipBorder:
      isDark
        ? "#465477"
        : "#d9e1ee",

    tooltipText:
      isDark
        ? "#ffffff"
        : "#17233d",

    tooltipLabel:
      isDark
        ? "#b8c4da"
        : "#637089",

    reference:
      isDark
        ? "#9aa7bd"
        : "#8c99ad",

    dotStroke:
      isDark
        ? "#18233f"
        : "#ffffff",

    legend:
      isDark
        ? "#aab5cb"
        : "#66748a",

  };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div
      className={`app ${theme}-theme`}
    >


      {/* =====================================================
          FLOATING NAVBAR
      ===================================================== */}

      <header className="navbar-shell">

        <nav className="topbar">


          <div className="brand">

            <div className="brand-mark">

              <img
                src="/alzora_favicon.png"
                alt="Alzora AI"
                className="brand-logo-image"
              />

            </div>


            <div className="brand-text">

              <h1>
                Alzora AI
              </h1>

              <p>
                Intelligent MRI Analysis
              </p>

            </div>

          </div>


          <div className="nav-center">

            <span className="nav-item active">
              MRI Analysis
            </span>

            <span className="nav-item">
              AI Research
            </span>

          </div>


          <div className="nav-actions">


            <button
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label="Toggle theme"
              title={
                isDark
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
            >

              <span className="theme-icon">

                {isDark
                  ? "☀"
                  : "☾"}

              </span>


              <span className="theme-label">

                {isDark
                  ? "Light"
                  : "Dark"}

              </span>

            </button>


            <div className="top-status">

              <span className="status-dot"></span>

              <span>
                AI MODEL ONLINE
              </span>

            </div>


          </div>

        </nav>

      </header>


      <main>


        {/* =====================================================
            HERO
        ===================================================== */}

        <section className="hero">


          <h2>

            Deep learning for early

            <br />

            <span>
              Alzheimer's MRI Analysis
            </span>

          </h2>


          <p>

            AI-assisted MRI detection and
            classification powered by
            MobileNetV2 and probability-based
            analysis with blockchain-backed
            data integrity.

          </p>


        </section>


        {/* =====================================================
            UPLOAD
        ===================================================== */}

        <section className="upload-panel">


          <div className="section-top">


            <div>

              <span className="eyebrow">
                MRI INPUT
              </span>

              <h3>
                Upload MRI Scan
              </h3>

            </div>


            <div className="secure-label">

              🔒 Secure Research Platform

            </div>


          </div>


          {!preview ? (

            <label className="dropzone">

              <input
                type="file"
                accept="image/*"
                onChange={handleFile}
              />


              <div className="upload-orb">
                ↑
              </div>


              <strong>
                Drop your MRI image here
              </strong>


              <span>
                or click to browse files
              </span>


              <small>
                JPG · JPEG · PNG · BMP · WEBP
              </small>

            </label>

          ) : (

            <div className="selected-file">


              <div className="mri-preview-wrap">

                <img
                  src={preview}
                  alt="MRI preview"
                />


                <div className="preview-overlay">
                  MRI PREVIEW
                </div>

              </div>


              <div className="file-details">

                <span className="eyebrow">
                  SELECTED SCAN
                </span>


                <h4>
                  {file?.name}
                </h4>


                <div className="file-meta">

                  <span>
                    {(file.size / 1024).toFixed(1)}
                    {" "}KB
                  </span>


                  <span>
                    {file.type || "Image"}
                  </span>

                </div>


                <button
                  className="secondary-btn"
                  onClick={reset}
                >
                  Choose another MRI
                </button>

              </div>


            </div>

          )}


          <button
            className="primary-btn"
            onClick={analyzeMRI}
            disabled={!file || loading}
          >


            {loading ? (

              <>

                <span className="spinner"></span>

                Alzora AI is analyzing...

              </>

            ) : (

              <>

                Analyze MRI

                <span className="arrow">
                  →
                </span>

              </>

            )}


          </button>


          {error && (

            <div className="error-box">

              ⚠️ {error}

            </div>

          )}


        </section>


        {/* =====================================================
            RESULTS
        ===================================================== */}

        {result && (

          <section
            id="results"
            className="results-section"
          >


            <div className="results-heading">


              <div>

                <span className="eyebrow">
                  AI INSIGHT REPORT
                </span>


                <h2>
                  Analysis Dashboard
                </h2>


                <p>
                  Model-generated analysis from the
                  uploaded MRI.
                </p>

              </div>


              <div className="complete-badge">

                <span>
                  ✓
                </span>

                ANALYSIS COMPLETE

              </div>


            </div>


            {/* =================================================
                DETECTION RESULT
            ================================================= */}

            <div
              className={`detection-card ${
                isDetected
                  ? "detection-positive"
                  : "detection-negative"
              }`}
            >


              <div className="detection-icon">

                {isDetected
                  ? "!"
                  : "✓"}

              </div>


              <div className="detection-content">

                <span className="eyebrow">
                  DEMENTIA DETECTION
                </span>


                <h2>

                  {isDetected
                    ? "DEMENTIA INDICATION DETECTED"
                    : "DEMENTIA INDICATION NOT DETECTED"}

                </h2>


                <p>
                  {detectionLabel}
                </p>

              </div>


              <div className="detection-score">

                <span>
                  Dementia Probability
                </span>


                <strong>
                  {dementiaProbability.toFixed(2)}%
                </strong>

              </div>


            </div>


            {/* =================================================
                PREDICTION + MRI
            ================================================= */}

            <div className="overview-grid">


              <div className="prediction-card">


                <div className="card-label-row">

                  <span className="eyebrow">
                    CLASSIFICATION RESULT
                  </span>


                  <span className="live-tag">
                    AI OUTPUT
                  </span>

                </div>


                <div className="prediction-icon">
                  🧠
                </div>


                <h2>
                  {prediction}
                </h2>


                <p>
                  Highest probability classification
                  generated by the MobileNetV2 model.
                </p>


                <div className="confidence-area">


                  <div className="confidence-header">

                    <span>
                      Model confidence
                    </span>


                    <strong>
                      {Number(
                        result.confidence
                      ).toFixed(2)}
                      %
                    </strong>

                  </div>


                  <div className="confidence-track">

                    <div
                      style={{
                        width: `${Math.min(
                          Number(
                            result.confidence
                          ),
                          100
                        )}%`,
                      }}
                    />

                  </div>


                </div>


              </div>


              <div className="mri-result-card">


                <div className="card-label-row">

                  <span className="eyebrow">
                    ANALYZED MRI
                  </span>


                  <span className="image-status">
                    READY
                  </span>

                </div>


                <div className="result-image-container">

                  <img
                    src={preview}
                    alt="Analyzed MRI"
                  />


                  <div className="image-hover">
                    MRI SCAN
                  </div>

                </div>


                <p>
                  {file?.name}
                </p>


              </div>


            </div>


            {/* =================================================
                PROBABILITY GRAPH
            ================================================= */}

            <div className="dashboard-card graph-card probability-profile-card">


              <div className="dashboard-card-header">


                <div>

                  <span className="eyebrow">
                    PREDICTION GRAPH
                  </span>


                  <h3>
                    Four-Class Probability Profile
                  </h3>


                  <p>
                    All four class lines are driven directly
                    by this MRI scan's model probabilities.
                    The predicted class is highlighted.
                  </p>

                </div>


                <div className="graph-badge">
                  25% REFERENCE
                </div>


              </div>


              {(() => {


                const wiggles = [

                  [
                    0,
                    -1.8,
                    2.4,
                    -1.2,
                    2.0,
                    -0.9,
                    0,
                  ],

                  [
                    0,
                    1.4,
                    -2.1,
                    1.7,
                    -1.4,
                    0.8,
                    0,
                  ],

                  [
                    0,
                    -1.1,
                    1.8,
                    -2.3,
                    1.2,
                    -0.7,
                    0,
                  ],

                  [
                    0,
                    2.0,
                    -1.5,
                    2.2,
                    -2.0,
                    1.0,
                    0,
                  ],

                ];


                const stages = [

                  "Reference",

                  "Signal 01",

                  "Signal 02",

                  "Signal 03",

                  "Signal 04",

                  "Signal 05",

                  "Current MRI",

                ];


                const profileData =
                  stages.map(
                    (
                      stage,
                      stageIndex
                    ) => {

                      const row = {
                        stage,
                      };


                      probabilities.forEach(
                        (
                          item,
                          classIndex
                        ) => {


                          const target =
                            Number(
                              item.value
                            ) || 0;


                          if (
                            stageIndex ===
                            stages.length - 1
                          ) {

                            row[
                              item.name
                            ] =
                              Number(
                                target.toFixed(2)
                              );

                            return;

                          }


                          const progress =
                            stageIndex /
                            (
                              stages.length - 1
                            );


                          const distance =
                            target - 25;


                          const amplitude =
                            Math.min(
                              3.2,
                              Math.max(
                                0.7,
                                Math.abs(
                                  distance
                                ) *
                                  0.045
                              )
                            );


                          const wiggle =
                            wiggles[
                              classIndex %
                                wiggles.length
                            ][
                              stageIndex
                            ] *
                            (
                              amplitude /
                              2.2
                            );


                          const value =
                            25 +
                            distance *
                              progress +
                            wiggle;


                          row[
                            item.name
                          ] =
                            Number(
                              Math.max(
                                0,
                                Math.min(
                                  100,
                                  value
                                )
                              ).toFixed(2)
                            );

                        }
                      );


                      return row;

                    }
                  );


                return (

                  <>


                    <ResponsiveContainer
                      width="100%"
                      height={460}
                    >

                      <LineChart
                        data={profileData}
                        margin={{
                          top: 45,
                          right: 45,
                          left: 5,
                          bottom: 25,
                        }}
                      >


                        <CartesianGrid
                          stroke={chart.grid}
                          strokeDasharray="3 6"
                          vertical={false}
                          opacity={0.45}
                        />


                        <XAxis
                          dataKey="stage"
                          tick={{
                            fill: chart.text,
                            fontSize: 10,
                          }}
                          axisLine={{
                            stroke: chart.axis,
                          }}
                          tickLine={{
                            stroke: chart.axis,
                          }}
                        />


                        <YAxis
                          domain={[
                            0,
                            100,
                          ]}
                          tickFormatter={(v) =>
                            `${v}%`
                          }
                          tick={{
                            fill: chart.text,
                            fontSize: 10,
                          }}
                          axisLine={{
                            stroke: chart.axis,
                          }}
                          tickLine={{
                            stroke: chart.axis,
                          }}
                        />


                        <Tooltip
                          contentStyle={{
                            background:
                              chart.tooltipBg,
                            border:
                              `1px solid ${chart.tooltipBorder}`,
                            borderRadius:
                              "12px",
                            color:
                              chart.tooltipText,
                            boxShadow:
                              isDark
                                ? "0 12px 30px rgba(0,0,0,.25)"
                                : "0 12px 30px rgba(30,50,90,.12)",
                          }}
                          labelStyle={{
                            color:
                              chart.tooltipLabel,
                            marginBottom:
                              "6px",
                          }}
                          itemStyle={{
                            color:
                              chart.tooltipText,
                          }}
                          formatter={(
                            value,
                            name
                          ) => [

                            `${Number(
                              value
                            ).toFixed(2)}%`,

                            name,

                          ]}
                        />


                        <ReferenceLine
                          y={25}
                          stroke={
                            chart.reference
                          }
                          strokeDasharray="7 7"
                          strokeWidth={1.3}
                          label={{
                            value:
                              "Equal-class reference · 25%",
                            position:
                              "insideTopRight",
                            fill:
                              chart.reference,
                            fontSize: 10,
                          }}
                        />


                        {probabilities.map(
                          (item) => {


                            const isPrediction =
                              item.name ===
                              prediction;


                            return (

                              <Line
                                key={
                                  item.name
                                }
                                type="linear"
                                dataKey={
                                  item.name
                                }
                                name={
                                  item.name
                                }
                                stroke={
                                  item.color
                                }
                                strokeWidth={
                                  isPrediction
                                    ? 4.5
                                    : 2.1
                                }
                                strokeOpacity={
                                  isPrediction
                                    ? 1
                                    : 0.68
                                }
                                dot={{
                                  r:
                                    isPrediction
                                      ? 4.5
                                      : 2.8,
                                  fill:
                                    item.color,
                                  stroke:
                                    chart.dotStroke,
                                  strokeWidth: 1.5,
                                }}
                                activeDot={{
                                  r: 7,
                                }}
                                isAnimationActive={
                                  true
                                }
                                animationDuration={
                                  900
                                }
                              >

                                <LabelList
                                  dataKey={
                                    item.name
                                  }
                                  position="top"
                                  content={({
                                    x,
                                    y,
                                    value,
                                    index,
                                  }) => {


                                    if (
                                      index !==
                                      profileData.length -
                                        1
                                    ) {

                                      return null;

                                    }


                                    return (

                                      <text
                                        x={x}
                                        y={
                                          y - 12
                                        }
                                        fill={
                                          item.color
                                        }
                                        textAnchor="middle"
                                        fontSize={12}
                                        fontWeight={
                                          isPrediction
                                            ? 800
                                            : 650
                                        }
                                      >

                                        {Number(
                                          value
                                        ).toFixed(
                                          1
                                        )}
                                        %

                                      </text>

                                    );

                                  }}
                                />

                              </Line>

                            );

                          }
                        )}


                      </LineChart>

                    </ResponsiveContainer>


                    <div className="profile-result-strip">


                      <div>

                        <span className="eyebrow">
                          CURRENT SCAN RESULT
                        </span>


                        <strong>
                          {prediction}
                        </strong>


                        <strong className="result-description">
                          - Highest model probability
                        </strong>

                      </div>


                      <div className="profile-confidence">

                        <span>
                          Probability :
                        </span>


                        <strong>
                          {predictionProbability.toFixed(
                            2
                          )}
                          %
                        </strong>

                      </div>


                      <div className="profile-probability-list">

                        {probabilities.map(
                          (item) => (

                            <div
                              key={item.name}
                              className={
                                `profile-probability-row ${
                                  item.name ===
                                  prediction
                                    ? "active-probability"
                                    : ""
                                }`
                              }
                            >

                              <span>
                                {item.name}
                              </span>


                              <span>
                                -
                              </span>


                              <strong>
                                {item.value.toFixed(
                                  2
                                )}
                                %
                              </strong>

                            </div>

                          )
                        )}

                      </div>


                    </div>


                    <div className="technical-note">

                      The final point of each curve is the
                      actual probability produced for this MRI.
                      The intermediate zig-zag points are a
                      visual probability trajectory and do not
                      represent additional scans, time progression,
                      or clinical measurements.

                    </div>


                  </>

                );

              })()}


            </div>


            {/* =================================================
                PIE + COMPARISON
            ================================================= */}

            <div className="dashboard-two">


              <div className="dashboard-card">


                <span className="eyebrow">
                  PROBABILITY DISTRIBUTION
                </span>


                <h3>
                  Model Output Distribution
                </h3>


                <p className="muted">
                  Relative probability contribution
                  across all four classes.
                </p>


                <ResponsiveContainer
                  width="100%"
                  height={340}
                >

                  <PieChart>


                    <Pie
                      data={probabilities}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="45%"
                      innerRadius={78}
                      outerRadius={120}
                      paddingAngle={4}
                    >

                      {probabilities.map(
                        (item, index) => (

                          <Cell
                            key={
                              item.name
                            }
                            fill={
                              COLORS[
                                index %
                                  COLORS.length
                              ]
                            }
                          />

                        )
                      )}

                    </Pie>


                    <Tooltip
                      contentStyle={{
                        background:
                          chart.tooltipBg,
                        border:
                          `1px solid ${chart.tooltipBorder}`,
                        borderRadius:
                          "10px",
                        color:
                          chart.tooltipText,
                      }}
                      formatter={(value) => [

                        `${Number(
                          value
                        ).toFixed(2)}%`,

                        "Probability",

                      ]}
                    />


                    <Legend
                      wrapperStyle={{
                        color:
                          chart.legend,
                        fontSize:
                          "11px",
                      }}
                    />


                  </PieChart>

                </ResponsiveContainer>


              </div>


              <div className="dashboard-card">


                <span className="eyebrow">
                  DETECTION ANALYSIS
                </span>


                <h3>
                  Detection vs Classification
                </h3>


                <p className="muted">
                  Model-output interpretation for the
                  uploaded MRI.
                </p>


                <div className="metric-list">


                  <div className="metric">

                    <span>
                      Detection status
                    </span>


                    <strong
                      className={
                        isDetected
                          ? "detection-text-positive"
                          : "detection-text-negative"
                      }
                    >
                      {detection}
                    </strong>

                  </div>


                  <div className="metric">

                    <span>
                      Classified class
                    </span>


                    <strong className="purple">
                      {prediction}
                    </strong>

                  </div>


                  <div className="metric">

                    <span>
                      Dementia probability
                    </span>


                    <strong>
                      {dementiaProbability.toFixed(
                        2
                      )}
                      %
                    </strong>

                  </div>


                  <div className="metric">

                    <span>
                      Non Demented probability
                    </span>


                    <strong>
                      {nonDementedProbability.toFixed(
                        2
                      )}
                      %
                    </strong>

                  </div>


                  <div className="metric highlight">

                    <span>
                      Classification probability
                    </span>


                    <strong>
                      {predictionProbability.toFixed(
                        2
                      )}
                      %
                    </strong>

                  </div>


                </div>


                <div className="technical-note">

                  Detection status is derived from the
                  four-class model output. Non Demented is
                  treated as the not-detected class, while
                  the three dementia-related classes are
                  treated as detected.

                </div>


              </div>


            </div>


            {/* =================================================
                BLOCKCHAIN VERIFICATION
            ================================================= */}

            {blockchain && (

              <div className="dashboard-card blockchain-card">


                <div className="dashboard-card-header">


                  <div>

                    <span className="eyebrow">
                      DATA INTEGRITY
                    </span>


                    <h3>
                      Blockchain Verification
                    </h3>


                    <p>
                      The MRI analysis record is linked to a
                      tamper-evident blockchain transaction.
                    </p>

                  </div>


                  <div
                    className={
                      blockchainConfirmed
                        ? "blockchain-status confirmed"
                        : "blockchain-status failed"
                    }
                  >

                    {blockchainConfirmed
                      ? "✓ VERIFIED"
                      : "⚠ NOT VERIFIED"}

                  </div>


                </div>


                <div className="blockchain-grid">


                  <div className="blockchain-item">

                    <span>
                      Record ID
                    </span>


                    <strong>
                      {blockchain.record_id ||
                        "N/A"}
                    </strong>

                  </div>


                  <div className="blockchain-item">

                    <span>
                      Blockchain Status
                    </span>


                    <strong>
                      {blockchain.status ||
                        "N/A"}
                    </strong>

                  </div>


                  <div className="blockchain-item">

                    <span>
                      Block Number
                    </span>


                    <strong>
                      {blockchain.block_number ??
                        "N/A"}
                    </strong>

                  </div>


                  <div className="blockchain-item">

                    <span>
                      Model Version
                    </span>


                    <strong>
                      {blockchain.model_version ||
                        "MobileNetV2"}
                    </strong>

                  </div>


                </div>


                <div className="hash-box">

                  <span>
                    MRI SHA-256 HASH
                  </span>


                  <code>
                    {blockchain.mri_hash ||
                      "Not available"}
                  </code>

                </div>


                <div className="hash-box">

                  <span>
                    TRANSACTION HASH
                  </span>


                  <code>
                    {blockchain.transaction_hash ||
                      "Not available"}
                  </code>

                </div>


                <div className="blockchain-note">

                  🔒 The MRI image itself is not stored on the
                  blockchain. A cryptographic SHA-256 hash is
                  recorded together with the AI prediction and
                  analysis metadata for integrity verification.

                </div>


              </div>

            )}


            {/* =================================================
                TABLE
            ================================================= */}

            <div className="dashboard-card table-card">


              <div className="dashboard-card-header">


                <div>

                  <span className="eyebrow">
                    DETAILED CLASSIFICATION
                  </span>


                  <h3>
                    Probability Breakdown
                  </h3>

                </div>


                <span className="output-tag">
                  LIVE MODEL OUTPUT
                </span>


              </div>


              <div className="table-wrapper">


                <table>


                  <thead>

                    <tr>

                      <th>
                        Rank
                      </th>

                      <th>
                        Classification
                      </th>

                      <th>
                        Probability
                      </th>

                      <th>
                        Distribution
                      </th>

                      <th>
                        Status
                      </th>

                    </tr>

                  </thead>


                  <tbody>


                    {ranked.map(
                      (
                        item,
                        index
                      ) => {


                        const isPrediction =
                          item.name ===
                          prediction;


                        return (

                          <tr
                            key={
                              item.name
                            }
                            className={
                              isPrediction
                                ? "prediction-row"
                                : ""
                            }
                          >


                            <td>

                              <span className="rank">
                                #{index + 1}
                              </span>

                            </td>


                            <td>

                              <div className="class-cell">

                                <i
                                  style={{
                                    background:
                                      item.color,
                                  }}
                                />


                                {item.name}

                              </div>

                            </td>


                            <td>

                              <strong>
                                {item.value.toFixed(
                                  2
                                )}
                                %
                              </strong>

                            </td>


                            <td>

                              <div className="mini-bar">

                                <span
                                  style={{
                                    width:
                                      `${Math.min(
                                        item.value,
                                        100
                                      )}%`,

                                    background:
                                      item.color,
                                  }}
                                />

                              </div>

                            </td>


                            <td>

                              {isPrediction ? (

                                <span className="predicted-tag">
                                  PREDICTED
                                </span>

                              ) : (

                                <span className="alternative-tag">
                                  ALTERNATIVE
                                </span>

                              )}

                            </td>


                          </tr>

                        );

                      }
                    )}


                  </tbody>


                </table>


              </div>


            </div>


            {/* =================================================
                INSIGHT
            ================================================= */}

            <div className="insight-card">


              <div className="insight-icon">
                ✦
              </div>


              <div>

                <span className="eyebrow">
                  AI ANALYSIS SUMMARY
                </span>


                <h3>
                  What did Alzora AI find?
                </h3>


                <p>

                  The model classified the uploaded MRI
                  as <strong>{prediction}</strong> with a
                  probability of{" "}

                  <strong>

                    {predictionProbability.toFixed(
                      2
                    )}

                    %

                  </strong>.

                </p>


                <p>

                  Detection status:
                  {" "}

                  <strong>
                    {detection}
                  </strong>

                  . The Non Demented class received{" "}

                  <strong>

                    {nonDementedProbability.toFixed(
                      2
                    )}

                    %

                  </strong>.

                </p>


              </div>


            </div>


            {/* =================================================
                MODEL
            ================================================= */}

            <div className="model-card">


              <div>

                <span className="eyebrow">
                  TECHNICAL PROFILE
                </span>


                <h3>
                  Alzora AI Model
                </h3>

              </div>


              <div className="model-grid">


                <div>

                  <span>
                    Architecture
                  </span>

                  <strong>
                    MobileNetV2
                  </strong>

                </div>


                <div>

                  <span>
                    Framework
                  </span>

                  <strong>
                    PyTorch
                  </strong>

                </div>


                <div>

                  <span>
                    Input
                  </span>

                  <strong>
                    224 × 224
                  </strong>

                </div>


                <div>

                  <span>
                    Classes
                  </span>

                  <strong>
                    4
                  </strong>

                </div>


                <div>

                  <span>
                    Detection
                  </span>

                  <strong>
                    Binary Interpretation
                  </strong>

                </div>


                <div>

                  <span>
                    Analysis
                  </span>

                  <strong>
                    MRI Based
                  </strong>

                </div>


              </div>


            </div>


            {/* =================================================
                DISCLAIMER
            ================================================= */}

            <div className="disclaimer">


              <div className="warning">
                ⚠️
              </div>


              <div>

                <strong>
                  Research Prototype — Not a Medical Diagnosis
                </strong>


                <p>

                  The displayed probabilities represent
                  model output for the uploaded image and
                  should not be interpreted as clinical
                  certainty or a substitute for professional
                  medical evaluation.

                </p>

              </div>


            </div>


            <button
              className="new-analysis"
              onClick={reset}
            >

              ← Analyze Another MRI

            </button>


          </section>

        )}


      </main>


      <footer>

        <strong>
          Alzora AI
        </strong>

        <span>
          {" "}•{" "}
        </span>

        Deep Learning MRI Research Prototype

      </footer>


    </div>

  );

}


export default App;