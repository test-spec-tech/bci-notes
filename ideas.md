# This was written by a LLM (claude), consider this content with skepticism

* Purpose: Test to see if AI can help with BCI research.

  
# Memory Enhancement BCI: Design Document

A multi-timescale, multi-modal brain-computer interface architecture for augmenting human episodic memory across encoding, consolidation, and retrieval.

---

## 1. Overview & Motivation

Human episodic memory is powerful but unreliable. We forget most of what we experience within hours, consolidation is fragile, and retrieval fails unpredictably. Existing memory BCIs target single mechanisms (e.g., replay detection or workload monitoring) but none address the full lifecycle of a memory trace from initial encoding through consolidation to retrieval.

This design proposes a **three-phase architecture** that intervenes at each stage of memory formation:

| Phase | Mechanism | Intervention |
|-------|-----------|-------------|
| **Encode** | Prediction-error detection + theta-gamma coupling | Boost attention and synaptic tagging at moments of high novelty |
| **Consolidate** | Sharp-wave ripple replay during sleep and rest | Selectively amplify content-decoded replays of target memories |
| **Retrieve** | Hippocampal index reactivation | Stimulate retrieval cues via cortical reinstatement patterns |

The system is designed in two hardware tiers (invasive and non-invasive) sharing a common software pipeline, allowing deployment across clinical and consumer contexts.

---

## 2. Neuroscience Foundations

### 2.1 Hippocampal Circuits

The hippocampus is the central hub for episodic memory. CA3 recurrent connections form an autoassociative network capable of pattern completion from partial cues ([Le Duigou et al., 2014](References.md); [Sammons et al., 2024](Memory.md); [Watson et al., 2024](Memory.md)). CA1 serves as the output stage, routing consolidated representations to neocortex via the subiculum and entorhinal cortex ([Takehara-Nishiuchi, 2014](Memory.md)).

**Design implication:** The system must monitor CA3/CA1 dynamics to detect encoding events and replay sequences. The CA3 autoassociative network is the primary target for prosthetic memory support.

### 2.2 Engrams and Memory Traces

Memory engrams are distributed ensembles of neurons whose reactivation drives recall ([Josselyn & Tonegawa, 2020](Memory.md); [Guskjolen & Cembrowski, 2023](Memory.md)). Engram cells are allocated during encoding via CREB-dependent mechanisms and are preferentially reactivated during retrieval. Recent work shows that astrocyte ensembles also participate in memory recall ([Williamson et al., 2025](Memory.md)).

**Design implication:** Rather than stimulating arbitrary neurons, interventions should target identified engram populations. Astrocytic calcium signals may serve as an additional readout of consolidation state.

### 2.3 Replay and Consolidation

Hippocampal replay compresses extended experiences into ~100ms sharp-wave ripple (SWR) events during sleep and quiet wakefulness ([Davidson et al., 2009](Memory.md); [Carr et al., 2011](Memory.md)). Replay can be forward or reverse ([Foster & Wilson, 2006](Memory.md)), occurs after even single experiences ([Berners-Lee et al., 2022](Memory.md)), and reflects specific past experiences rather than future plans ([Gillespie et al., 2021](Memory.md)).

Cortical replay of spiking sequences occurs during human memory retrieval ([Vaz et al., 2020](Memory.md)), with high-gamma reinstatement in temporal and prefrontal cortex ([Yaffe et al., 2017](Memory.md)).

**Design implication:** The consolidation module must detect SWRs in real-time, decode their content, and selectively boost replays of target memories while leaving others undisturbed.

### 2.4 Spatial Scaffolds and Cognitive Maps

Grid cells in the entorhinal cortex form toroidal population manifolds ([Gardner et al., 2022](Memory.md)) that provide a metric scaffold for both spatial and non-spatial memories ([Chandra et al., 2025](Memory.md)). Vector coding and place coding share common directional signals ([Zhou et al., 2024](Memory.md)), and theta sweeps alternate between hemispheres during navigation ([Vollan et al., 2025](Memory.md)).

**Design implication:** Grid cell scaffolds can be leveraged as "memory palaces" - artificial spatial frameworks onto which non-spatial memories are mapped, exploiting the brain's evolved spatial memory infrastructure.

### 2.5 Prefrontal-Hippocampal Interactions

Prefrontal cortex drives memory recall through feature representations ([Yadav et al., 2022](Memory.md)) and interacts with the medial temporal lobe for long-term storage ([Simons & Spiers, 2003](Memory.md)). Working memory maintenance depends on prefrontal circuitry ([Lara & Wallis, 2015](Memory.md)) while the ventromedial prefrontal cortex integrates memory with decision-making and schemas ([Weilbacher & Gluth, 2016](Memory.md); [Hebscher & Gilboa, 2016](Memory.md)).

**Design implication:** Retrieval assistance must engage prefrontal-hippocampal dialogue. The system can prime prefrontal feature representations to facilitate hippocampal pattern completion.

### 2.6 Oscillatory Signatures

Memory operations have distinct oscillatory signatures:

| Oscillation | Frequency | Role in Memory |
|-------------|-----------|---------------|
| Theta | 4-8 Hz | Encoding gate; phase-locks new information |
| Gamma | 30-100 Hz | Feature binding; content representation |
| Theta-gamma PAC | Theta phase / gamma amplitude | Organizes multi-item sequences in working memory |
| Sharp-wave ripples | 150-250 Hz | Replay and consolidation during sleep/rest |
| High gamma | 70-150 Hz | Cortical reinstatement during retrieval |
| Slow oscillations | 0.5-1 Hz | Sleep-stage gating of consolidation |

Cortical oscillations show a spectrolaminar motif across primate cortex ([Mendoza-Halliday et al., 2024](Memory.md)), and oscillatory rhythmicity itself carries information about cortical architecture ([Myrov et al., 2024](Memory.md)).

---

## 3. System Architecture

The system operates in three phases aligned with the natural memory lifecycle:

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ENHANCEMENT BCI                    │
│                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  ENCODE  │────▶│ CONSOLIDATE  │────▶│   RETRIEVE   │    │
│  │          │     │              │     │              │    │
│  │ Detect   │     │ Decode SWR   │     │ Prime cue    │    │
│  │ novelty  │     │ content      │     │ patterns     │    │
│  │ Boost    │     │ Boost target │     │ Facilitate   │    │
│  │ tagging  │     │ replays      │     │ completion   │    │
│  └──────────┘     └──────────────┘     └──────────────┘    │
│        │                 │                    │             │
│        └─────────────────┼────────────────────┘             │
│                          ▼                                  │
│                 ┌────────────────┐                          │
│                 │  MEMORY INDEX  │                          │
│                 │  (Compressed   │                          │
│                 │   pointers)    │                          │
│                 └────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Phase 1: Encoding Optimization

**Goal:** Identify high-value encoding moments and strengthen synaptic tagging.

| Component | Function |
|-----------|----------|
| Prediction-error detector | Monitors mismatch between expected and actual neural activity (a proxy for novelty/surprise) |
| Theta-gamma PAC tracker | Measures phase-amplitude coupling as an index of active encoding |
| Synaptic tagging window | Detects the ~1-2 hour post-encoding window during which tagged synapses can capture plasticity-related proteins |
| Stimulation controller | Delivers timed neuromodulatory input (dopaminergic/noradrenergic) to strengthen tagging |

**Mechanism:** When the prediction-error signal exceeds threshold during strong theta-gamma coupling, the system marks this as an encoding event. Within the subsequent synaptic tagging window (~30-120 min), it delivers low-intensity stimulation to promote protein synthesis-dependent consolidation at the tagged synapses. This exploits the synaptic tagging and capture (STC) hypothesis: weak synaptic changes can be stabilized if plasticity-related proteins arrive within a critical window.

### 3.2 Phase 2: Consolidation Boosting

**Goal:** Selectively strengthen target memories during sleep and quiet rest.

| Component | Function |
|-----------|----------|
| SWR detector | Real-time detection of sharp-wave ripple events (150-250 Hz) |
| Content decoder | Classifies replay content using pre-trained neural decoders |
| Replay booster | Closed-loop stimulation timed to SWR events containing target content |
| Sleep stage monitor | Tracks slow oscillations and spindles to time interventions to NREM sleep |
| TMR controller | Delivers auditory/olfactory cues during slow-wave sleep to bias replay toward target memories |

**Mechanism:** Rather than blindly amplifying all SWRs (which could strengthen unwanted memories or interfere with natural forgetting), the system decodes replay content in real-time. Only replays matching target memory signatures receive boosting stimulation. This is combined with targeted memory reactivation (TMR) - presenting learning-associated sensory cues during NREM sleep to bias which memories get replayed.

The system also coordinates with the slow oscillation up-state and sleep spindles (the SO-spindle-ripple coupling sequence) to ensure stimulation arrives at the optimal phase for synaptic consolidation.

### 3.3 Phase 3: Retrieval Assistance

**Goal:** Facilitate recall by priming hippocampal index codes and cortical reinstatement.

| Component | Function |
|-----------|----------|
| Context detector | Identifies retrieval attempts from prefrontal activation patterns |
| Index activator | Stimulates hippocampal index codes to initiate pattern completion |
| Reinstatement monitor | Tracks cortical high-gamma reinstatement to verify successful retrieval |
| Confidence estimator | Assesses retrieval strength and provides graded assistance |

**Mechanism:** The hippocampus stores compressed "pointers" (index codes) that, when reactivated, drive reinstatement of distributed cortical representations. The system detects retrieval attempts, identifies the target memory from partial cues, and provides graded stimulation to the hippocampal index. This is more efficient than trying to reactivate the full distributed trace directly - manipulate the pointer, not the data.

---

## 4. Hardware Tiers

### 4.1 Tier 1: Invasive (Clinical/Research)

For participants with clinical indications (e.g., epilepsy patients with depth electrodes, or future implant recipients).

| Component | Specification |
|-----------|--------------|
| Recording | High-density Utah arrays or Neuropixels in hippocampus (CA1/CA3), entorhinal cortex, prefrontal cortex |
| Stimulation | Microstimulation via same electrode arrays; temporal interference (TI) for deeper targets |
| Processing | On-implant neuromorphic chip for SWR detection and spike sorting ([Liu et al., 2025](References.md)); memristor-based adaptive decoding |
| Communication | Wireless via inductive link ([Won et al., 2023](References.md)) to external processing unit |
| Power | Wireless power transfer; target <50mW for implanted components |

This tier leverages the BRAND platform for real-time closed-loop processing ([Ali et al., 2024](References.md)), which provides the infrastructure for asynchronous neural decoding with deep network models.

### 4.2 Tier 2: Non-Invasive (Consumer/Wellness)

For healthy individuals seeking memory enhancement.

| Component | Specification |
|-----------|--------------|
| Recording | High-density EEG (64-256 channels) + fNIRS for hemodynamic confirmation |
| Stimulation | Transcranial temporal interference stimulation (tTIS) for focal deep-brain targeting; transcranial alternating current stimulation (tACS) for oscillatory entrainment |
| Processing | Edge compute module (wearable) running compressed decoder models |
| Sensors | Wrist-worn autonomic monitoring (heart rate variability, skin conductance) for arousal/sleep staging |
| Form factor | Behind-ear EEG + forehead patch; sleep headband variant |

This tier trades spatial resolution for accessibility. It cannot decode individual replay content but can:
- Detect SWR-band power increases and overall consolidation state
- Entrain theta-gamma coupling during encoding via tACS
- Deliver TMR cues during detected NREM sleep
- Monitor working memory load in real-time ([Mora-Sanchez et al., 2020](References.md); [Asgher et al., 2020](References.md))

### 4.3 Shared Software Architecture

Both tiers run the same software stack, differing only in signal quality and available interventions:

```
┌──────────────────────────────────────┐
│         Application Layer            │
│  (Memory journal, learning goals,    │
│   consolidation scheduling)          │
├──────────────────────────────────────┤
│         Decoder Layer                │
│  (Neural state classification,       │
│   replay content decoding,           │
│   sleep staging)                     │
├──────────────────────────────────────┤
│         Signal Processing Layer      │
│  (Filtering, artifact rejection,     │
│   oscillation extraction, SWR det.)  │
├──────────────────────────────────────┤
│         Hardware Abstraction Layer   │
│  (Tier-agnostic interface to         │
│   recording and stimulation)         │
└──────────────────────────────────────┘
```

---

## 5. Software Pipeline

### 5.1 Real-Time Signal Processing

| Stage | Latency Budget | Method |
|-------|---------------|--------|
| Bandpass filtering | <1 ms | IIR filters; separate theta (4-8 Hz), gamma (30-100 Hz), ripple (150-250 Hz) bands |
| Artifact rejection | <2 ms | Template matching for movement/EMG artifacts; adaptive thresholding |
| SWR detection | <10 ms | Ripple-band power threshold + duration criterion (>25 ms); validated against offline detection |
| Spike sorting | <5 ms | Neuromorphic on-chip sorting (Tier 1); not available in Tier 2 |
| Phase estimation | <5 ms | Hilbert transform on buffered theta signal for PAC computation |

**Total closed-loop latency target:** <25 ms (Tier 1), <50 ms (Tier 2)

### 5.2 Neural Decoders

| Decoder | Input | Output | Architecture |
|---------|-------|--------|-------------|
| Encoding state | Theta-gamma PAC + prediction error | Binary: encoding/not-encoding | Lightweight CNN on spectrogram features |
| Replay content | CA1 population vectors during SWR | Memory identity (from learned library) | LFADS-based latent factor model ([Ali et al., 2024](References.md)) |
| Sleep stage | EEG slow oscillations + spindles + EMG | Wake/N1/N2/N3/REM | Random forest on spectral features |
| Retrieval state | PFC activation + hippocampal theta | Retrieval attempt + target memory | Sequence-to-sequence model on neural trajectories |
| Working memory load | Frontal theta + parietal alpha | Load level (0-7 items) | LSTM ([Asgher et al., 2020](References.md)) |

All decoders use a calibration session to learn participant-specific neural signatures, then adapt online via memristor-based co-evolutionary learning ([Liu et al., 2025](References.md)).

### 5.3 Stimulation Protocols

| Protocol | Target | Parameters | Phase |
|----------|--------|-----------|-------|
| Theta entrainment | Hippocampus | 6 Hz tACS, 1-2 mA | Encoding |
| Gamma burst | Entorhinal-hippocampal | 40 Hz bursts locked to theta trough | Encoding |
| SWR boost | CA1 | Single-pulse microstim at ripple onset, 50-100 uA | Consolidation |
| SO up-state targeting | Frontal cortex | tDCS pulse at SO up-state, 0.5 mA | Consolidation |
| TMR cue | Auditory/olfactory | Learning-associated sensory cue during N3 | Consolidation |
| Index reactivation | CA3 | Patterned microstim matching index code, 20-50 uA | Retrieval |

---

## 6. Novel Design Elements

### 6.1 Hippocampal Indexing as Compression

Rather than attempting to read or write full memory traces (which are distributed across cortex), the system treats hippocampal representations as **compressed pointers**. This is inspired by hippocampal indexing theory: the hippocampus stores a sparse index code that, when reactivated, reinstates the full distributed cortical pattern.

**Advantage:** The system only needs to decode and manipulate a low-dimensional index space (~100-1000 dimensions) rather than the full cortical representation. This dramatically reduces the decoder complexity and stimulation precision required.

### 6.2 Synaptic Tagging Window Exploitation

The synaptic tagging and capture (STC) mechanism creates a ~1-2 hour window after initial encoding during which weak synaptic traces can be stabilized by plasticity-related proteins. The system detects encoding events and schedules neuromodulatory stimulation within this window.

**Protocol:**
1. Detect encoding event (prediction error + theta-gamma PAC)
2. Log timestamp and estimated trace strength
3. Within 30-120 minutes, deliver dopaminergic/noradrenergic stimulation to promote protein synthesis
4. Verify consolidation via subsequent replay detection

This exploits a natural biological mechanism that is normally governed by behavioral state (arousal, reward) but can be artificially triggered.

### 6.3 Content-Decoded Replay Boosting

Existing closed-loop SWR systems boost all detected ripples indiscriminately. This design adds a **content decoder** that classifies replay content before deciding whether to boost.

**Why this matters:**
- Not all replays are beneficial (some may maintain traumatic or irrelevant memories)
- Natural forgetting serves a function; indiscriminate boosting could cause interference
- Users can specify which memories to prioritize for consolidation

### 6.4 Grid Cell Memory Palace

Grid cells provide a metric scaffold for both spatial and conceptual representations ([Gardner et al., 2022](Memory.md); [Chandra et al., 2025](Memory.md)). The system creates artificial "memory palaces" by:

1. During encoding, pairing new information with spatial trajectories through a virtual environment
2. Stimulating grid cell patterns corresponding to specific locations during consolidation
3. During retrieval, reactivating the spatial scaffold to provide additional retrieval cues

This leverages the method of loci (a mnemonic technique dating to ancient Greece) with direct neural implementation rather than relying on voluntary imagination.

### 6.5 Predictive Coding for Novelty Detection

The system monitors prediction error signals as a natural index of information novelty. High prediction error indicates surprising, potentially important information that warrants enhanced encoding. Low prediction error indicates redundant input that can be safely compressed.

This uses the brain's own computational framework (predictive coding across the cortical hierarchy; [Felleman & Van Essen, 1991](Memory.md); [Rvachev, 2024](Memory.md)) as a relevance signal, rather than requiring the user to manually tag important moments.

### 6.6 Multi-Timescale Integration

The architecture uniquely spans three timescales:

| Timescale | Duration | Process | System Action |
|-----------|----------|---------|--------------|
| Fast | 10-100 ms | SWR replay, gamma bursts | Real-time closed-loop stimulation |
| Medium | Minutes to hours | Synaptic tagging window, working memory | Scheduled neuromodulatory support |
| Slow | Hours to days | Systems consolidation, sleep cycles | Overnight consolidation scheduling, TMR |

No existing system integrates across all three timescales with a unified memory model.

---

## 7. Risk Analysis & Open Questions

### 7.1 Safety Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Seizure induction from closed-loop stimulation | High | Afterdischarge detection with immediate stimulation halt; conservative current limits; epileptologist oversight |
| Unwanted memory strengthening (e.g., traumatic content) | Medium | Content decoder acts as filter; user blacklist for memory categories; default-off for emotionally tagged events |
| Interference with natural forgetting | Medium | Selective boosting (not blanket amplification); periodic "natural consolidation" windows without intervention |
| Long-term neural adaptation reducing efficacy | Medium | Adaptive stimulation parameters; periodic recalibration; stimulation holidays |
| Tissue damage from chronic implant (Tier 1) | High | Biocompatible materials; wireless power to minimize implant size; regular impedance monitoring |

### 7.2 Ethical Considerations

| Issue | Consideration |
|-------|--------------|
| Memory authenticity | Enhanced memories may feel more vivid than warranted by the original experience; users must be informed |
| Consent for consolidation targets | Who decides which memories get boosted? Must remain under user control |
| Cognitive liberty | Potential for coercive use (e.g., employer-mandated memory enhancement); requires regulatory framework |
| Equity of access | Tier 1 is surgical; Tier 2 is consumer. Performance gap creates potential inequality |
| Data privacy | Neural recordings contain intimate cognitive data; requires on-device processing and strong encryption |

### 7.3 Open Technical Questions

| Question | Current Status | Path Forward |
|----------|---------------|-------------|
| Can replay content be decoded with sufficient accuracy for selective boosting? | Demonstrated in rodents; limited human data ([Vaz et al., 2020](Memory.md)) | Piggyback on epilepsy monitoring studies for human validation |
| Does tTIS achieve sufficient focal depth for hippocampal targeting? | Promising in simulation and early trials | Requires systematic dose-finding studies |
| How long does the synaptic tagging window last in humans? | ~1-2 hours in rodent slice preparations | Human studies with behavioral + neural markers needed |
| Can theta-gamma PAC be reliably entrained non-invasively? | Inconsistent results with tACS alone | Combine tACS with behavioral task design; closed-loop phase tracking |
| Will chronic replay boosting cause runaway potentiation? | Unknown | Long-term animal studies with consolidation monitoring; homeostatic mechanisms |
| How does the system handle overlapping memory traces? | Untested | Pattern separation metrics as quality control; user disambiguation interface |

---

## 8. References

This design draws on papers catalogued in this repository:

- **[References.md](References.md)** - BCI platforms, neural decoding, speech neuroprostheses, neuromorphic hardware, memory-BCI interfaces, and workload monitoring
- **[Memory.md](Memory.md)** - Hippocampal circuits, engrams, replay dynamics, grid cells, spatial memory, prefrontal-hippocampal interactions, oscillations, and cortical organization

### Key papers by topic

| Topic | Key References |
|-------|---------------|
| Hippocampal circuits | Le Duigou et al. 2014; Sammons et al. 2024; Watson et al. 2024 |
| Engrams | Josselyn & Tonegawa 2020; Guskjolen & Cembrowski 2023; Williamson et al. 2025 |
| Replay | Davidson et al. 2009; Carr et al. 2011; Vaz et al. 2020; Berners-Lee et al. 2022; Gillespie et al. 2021 |
| Spatial scaffolds | Gardner et al. 2022; Chandra et al. 2025; Sargolini et al. 2006; Zhou et al. 2024 |
| Prefrontal memory | Yadav et al. 2022; Simons & Spiers 2003; Lara & Wallis 2015 |
| Oscillations | Mendoza-Halliday et al. 2024; Myrov et al. 2024 |
| Memory circuits | Ferguson et al. 2019; Raslau et al. 2015; Tingley & Buzsaki 2018 |
| BCI platforms | Ali et al. 2024 (BRAND); Liu et al. 2025 (memristor decoder) |
| Wireless/implant | Won et al. 2023 |
| Memory BCI | Burke et al. 2015; Mora-Sanchez et al. 2020; Asgher et al. 2020 |
| Cortical organization | Felleman & Van Essen 1991; Pang et al. 2023; Rvachev 2024 |

### Additional sources informing the design

- **MIMO hippocampal prosthesis:** Berger, Song, et al. - multi-input multi-output model replacing damaged hippocampal circuitry
- **Synaptic tagging and capture:** Frey & Morris, 1997 - the STC hypothesis for late-phase LTP
- **Targeted memory reactivation:** Rasch et al., 2007 - odor cues during sleep boost declarative memory
- **Temporal interference stimulation:** Grossman et al., 2017 - non-invasive deep brain stimulation via interfering electric fields
- **Predictive coding:** Rao & Ballard, 1999; Friston, 2005 - hierarchical prediction error as a cortical organizing principle


# Memory Writing BCI: Design Document

A brain-computer interface architecture for writing artificial memories — implanting experiences, knowledge, and skills that were never naturally encoded — via targeted neural pattern injection, integration, and verification.

---

## 1. Overview & Motivation

Memory enhancement boosts natural encoding, consolidation, and retrieval. But enhancement cannot help when there is **no natural encoding event to boost**. Anterograde amnesia patients cannot form new memories. Skill acquisition requires months of practice. Traumatic memories resist overwriting. These problems require a fundamentally different approach: **writing memories from scratch**.

Memory writing must solve five problems that enhancement never faces:

| Stage | Problem | Approach |
|-------|---------|----------|
| **Specify** | Content must be externally defined in a machine-readable format | Memory descriptor language with associative link maps |
| **Generate** | Content must be translated into participant-specific neural patterns | Manifold-constrained content-to-engram encoder |
| **Inject** | Patterns must be delivered to the correct neurons at the correct time | Multi-site patterned microstimulation with phase-precise timing |
| **Integrate** | The artificial trace must link to existing memory networks | Artificial replay injection during NREM sleep; schema-primed consolidation |
| **Verify** | The written memory must be confirmed accurate and non-destructive | Retrieval probe with content decoding and fidelity scoring |

Clinical applications include prosthetic memory for hippocampal damage ([Burke et al., 2015](References.md)), PTSD memory replacement via reconsolidation editing, accelerated skill acquisition, and educational content delivery. More broadly, neuromodulation for neurological and psychiatric disorders has matured as a field ([Johnson et al., 2013](References.md)), and memory writing represents its logical extension — from modulating ongoing neural dynamics to specifying new ones. The feasibility of artificial engram creation has been demonstrated in rodents, where false place memories were optogenetically implanted in hippocampal ensembles ([Ramirez et al., 2013](#9-references)), and memory engram theory provides the conceptual framework ([Josselyn & Tonegawa, 2020](Memory.md)).

A central insight of this design is that memory writing has **two modes**, not one:

| Mode | Mechanism | Best For |
|------|-----------|----------|
| **De novo writing** | Create new engram in unoccupied neural territory | Novel content with no existing trace |
| **Reconsolidation editing** | Reactivate existing memory → destabilize → inject modified content | Updating, correcting, or replacing existing memories |

---

## 2. Neuroscience Foundations

### 2.1 Engram Theory, CREB Allocation & Neurogenesis

Memory engrams are sparse, distributed neuron ensembles whose reactivation drives recall ([Josselyn & Tonegawa, 2020](Memory.md); [Guskjolen & Cembrowski, 2023](Memory.md)). During natural encoding, neurons with high CREB levels are preferentially recruited into engrams — a competitive allocation process ([Han et al., 2007](#9-references)). Engram membership is not random; it is governed by the excitability state of neurons at the time of encoding.

Critically, artificial manipulation of CREB levels can bias which neurons join an engram, and optogenetic reactivation of engram-allocated neurons is sufficient to drive recall even for events that never occurred. Ramirez et al. (2013) demonstrated this by labeling place-cell ensembles active in context A, then reactivating them during fear conditioning in context B, creating a false fear memory for context A ([Ramirez et al., 2013](#9-references)).

A particularly important class of allocatable neurons are **adult-born granule cells** in the dentate gyrus. During their critical period (~4-6 weeks post-mitosis), newborn DGCs exhibit lower LTP thresholds, higher intrinsic excitability, and reduced sensitivity to GABAergic inhibition ([Aimone et al., 2011](#9-references)). Increasing adult neurogenesis is sufficient to improve pattern separation ([Sahay et al., 2011](#9-references)). These properties make newborn neurons natural "blank slates" — uncommitted to existing engrams and primed for plasticity.

**Design implication:** The memory writing system should preferentially target two neuron populations: (1) high-CREB mature neurons identified via excitability profiling, and (2) adult-born DG granule cells in their critical period. Newborn neurons are ideal write targets because they are not yet committed to existing engrams (reducing overwrite risk) and require less stimulation energy to induce plasticity. The system should track neurogenesis markers to identify optimal write windows. Writing to mature, low-excitability neurons will produce weak, unstable traces.

### 2.2 CA3 Autoassociation & Pattern Completion

CA3 recurrent connections form an autoassociative network: partial input patterns are completed to full stored attractors ([Le Duigou et al., 2014](Memory.md); [Sammons et al., 2024](Memory.md); [Watson et al., 2024](Memory.md)). Human CA3 uses specific functional connectivity rules that make this completion highly efficient — approximately 20-30% of the pattern is sufficient to drive convergence to the full attractor.

However, CA3 does not simply complete patterns. The dentate gyrus upstream performs **pattern separation** — orthogonalizing similar inputs before they reach CA3 — and the balance between separation (DG) and completion (CA3) is regulated by neuromodulatory state and the relative contribution of newborn vs. mature granule cells. A written pattern that bypasses DG and enters CA3 directly will encounter the completion machinery without the orthogonalization step, risking convergence to an existing attractor rather than formation of a new one.

**Design implication:** The system can exploit CA3 pattern completion by writing only a **seed pattern** (20-30% of the target engram) and allowing autoassociative dynamics to fill in the remainder. But the seed must be sufficiently orthogonal to existing attractors — a condition that DG normally ensures but that artificial injection must verify computationally. The pattern generator must simulate attractor dynamics against the participant's known memory landscape before injection. If the seed falls within the basin of attraction of an existing engram, the write will silently overwrite or blend with that memory.

### 2.3 Hippocampal Indexing & Schema-Accelerated Consolidation

The hippocampus stores compressed index codes — sparse pointers (~100-1000 dimensions) that, when reactivated, reinstate distributed cortical representations. The full richness of a memory lives in cortex, not hippocampus. The hippocampal trace is a lookup key.

The standard model of systems consolidation holds that hippocampal-to-cortical transfer takes weeks. But this timeline is **dramatically compressed when new content matches an existing cortical schema**. Tse et al. (2007) showed that rats with an established spatial schema could consolidate new paired associations within 48 hours — compared to weeks without a schema ([Tse et al., 2007](#9-references)). The mechanism involves immediate neocortical gene activation (zif268, Arc) at encoding time when schemas exist ([Tse et al., 2011](#9-references)), bypassing the slow offline consolidation normally required.

**Design implication:** The system should implement two consolidation strategies: (1) **Schema-absent consolidation** — write the hippocampal index, then drive artificial replay over multiple nights to gradually build the cortical trace. (2) **Schema-primed consolidation** — when the written content maps onto an existing cortical schema, write the hippocampal index and deliver a single concentrated replay session. Schema-primed writes may consolidate in hours rather than days. The content specification module must assess schema compatibility and route memories to the appropriate consolidation pathway. This is the single largest efficiency gain available: schema-compatible memories are 10-100x faster to consolidate.

### 2.4 Replay, Sequence Ordering & Preplay

Sharp-wave ripple (SWR) replay compresses experiences into ~100ms bursts during sleep and quiet rest ([Davidson et al., 2009](Memory.md); [Carr et al., 2011](Memory.md)). Replay occurs after even single experiences ([Berners-Lee et al., 2022](Memory.md)) and drives cortical spiking reinstatement during retrieval ([Vaz et al., 2020](Memory.md)).

Critically, replay is not a monolithic phenomenon. **Forward replay** (events replayed in experienced order) dominates during quiet wakefulness and is associated with planning and evaluation. **Reverse replay** (events replayed backwards) dominates during reward-associated pauses and is associated with credit assignment and value updating ([Foster & Wilson, 2006](Memory.md)). The temporal ordering of cells within a ripple determines the content and function of the replay event.

Furthermore, the hippocampus generates **preplay** — sequences representing trajectories not yet experienced. Preplay suggests the hippocampal network spontaneously explores its state space, generating candidate experiences that can be "confirmed" or "rejected" by subsequent real experience.

**Design implication:** The artificial replay engine must control **sequence direction and content**, not merely trigger SWR-like bursts. For episodic memories (sequences of events), forward replay preserves narrative order and supports later sequential retrieval. For associative memories (causal relationships, skill-reward links), reverse replay from outcome to cause may be more effective for consolidation. The system should deliver a mixture of forward and reverse replays, with the ratio determined by memory type. Preplay could also be exploited: rather than writing a memory de novo, the system could detect spontaneous preplay sequences that approximate the target content and selectively reinforce them, converting exploration into memory.

### 2.5 Synaptic Plasticity & Tagging

Long-term memory formation requires protein synthesis-dependent synaptic modification. The synaptic tagging and capture (STC) mechanism creates a ~1-2 hour window: initial stimulation creates "tags" at activated synapses, and plasticity-related proteins (PRPs) must arrive within this window to stabilize the trace ([Kennedy, 2013](Memory.md).

For memory writing, pattern injection creates the tags, but PRP delivery is not guaranteed — the brain's natural neuromodulatory systems (dopamine, norepinephrine) may not engage because no behaviorally relevant event occurred.

**Design implication:** The injection protocol must be paired with neuromodulatory support. Options include: (1) pharmacological priming (low-dose dopamine agonist to raise baseline PRP availability), (2) electrical co-stimulation of VTA/LC neuromodulatory nuclei, or (3) pairing injection with a behaviorally salient event (e.g., reward signal). The STC window constrains timing: all injection and stabilization must complete within ~1-2 hours. Multiple writing sessions may be needed for complex memories, with each session targeting a subset of the engram.

### 2.6 Spatial & Contextual Scaffolds

Grid cells in entorhinal cortex provide toroidal metric scaffolds ([Gardner et al., 2022](Memory.md)) that organize both spatial and non-spatial memories ([Chandra et al., 2025](Memory.md)). Every episodic memory has spatial and contextual components — even abstract knowledge is encoded relative to a spatial/temporal framework.

For written memories, this context must be supplied artificially. A memory without spatial context will lack the scaffolding that makes natural memories navigable and retrievable.

**Design implication:** The memory specification must include a spatial/contextual scaffold. The system can leverage the grid cell "memory palace" approach: assign each written memory a location within a virtual spatial framework, and co-activate the corresponding grid cell patterns during injection. The toroidal topology of grid cell manifolds ([Gardner et al., 2022](Memory.md)) means that relational structure (not just spatial location) can be encoded — memories that are "conceptually nearby" can be assigned adjacent positions on the torus, creating a navigable semantic topology. This goes beyond the classical method of loci: it is a general-purpose relational scaffold.

### 2.7 Inhibitory Sculpting & Disinhibition Gates

Memory engrams are defined not only by which excitatory neurons fire, but by which are **prevented** from firing. Inhibitory interneurons play three distinct roles in engram formation:

- **PV+ (parvalbumin) interneurons** provide fast perisomatic inhibition that regulates the temporal precision of pyramidal cell firing. They define the temporal window within which engram neurons must co-activate to form Hebbian associations.
- **SST+ (somatostatin) interneurons** target dendrites and regulate the size of the engram through lateral inhibition. Silencing DG SST+ interneurons during encoding increases engram size; activating them shrinks it ([Stefanelli et al., 2016](#9-references)). This is competitive exclusion: active engram cells recruit SST+ interneurons that suppress their neighbors.
- **VIP+ (vasoactive intestinal peptide) interneurons** inhibit SST+ and PV+ interneurons, creating **disinhibitory windows** that permit plasticity in target pyramidal cells ([Letzkus et al., 2015](#9-references)). Salient events trigger VIP+ activation, which releases projection neurons from inhibition.

Lovett-Barron et al. (2014) showed that during fear conditioning, cholinergic input activates dendrite-targeting interneurons in CA1, gating which sensory information is incorporated into the memory trace ([Lovett-Barron et al., 2014](#9-references)).

**Design implication:** Memory writing cannot simply activate the target excitatory neurons — it must simultaneously engineer the appropriate inhibitory landscape. This means: (1) co-activating VIP+ interneurons to open a disinhibitory plasticity window at the injection site, (2) allowing SST+ interneurons to constrain engram size to natural proportions (preventing pathologically large engrams that degrade pattern separation), and (3) timing injection to PV+ interneuron rhythms that gate the temporal precision of Hebbian binding. Ignoring the inhibitory context will produce engrams with wrong size, wrong temporal precision, and wrong integration with surrounding networks.

### 2.8 Dendritic Computation & Branch-Specific Plasticity

Memory traces are not stored at the level of whole neurons — they are stored at **individual dendritic branches**. Each pyramidal neuron has dozens of branches, and different branches can participate in different memory traces through local NMDA spikes, branch-specific calcium events, and clustered synaptic plasticity.

Govindarajan et al. (2006) proposed that translation-dependent plasticity at synapses clustered within a dendritic branch is the elementary unit of long-term memory storage ([Govindarajan et al., 2006](#9-references)). Cichon & Bhatt (2015) demonstrated this directly: different motor learning tasks induce dendritic calcium spikes on different apical tuft branches of the same layer V neuron, with branch-specific potentiation persisting for days ([Cichon & Bhatt, 2015](#9-references)). Kastellakis et al. (2015) showed that synaptic clustering within dendrites dramatically increases memory capacity and enables feature binding — related features of a memory are stored on the same branch ([Kastellakis et al., 2015](#9-references)).

**Design implication:** The standard model of memory writing — "activate neuron X" — is too coarse. True high-fidelity writing requires engaging specific **dendritic branches** within target neurons, activating clustered synapses that bind the correct features together. This has profound hardware implications: electrode or optogenetic resolution must reach the dendritic scale (~1-10 μm), not merely the somatic scale (~10-50 μm). In the near term, somatic-resolution writing is feasible but will produce low-resolution memories (correct gist, imprecise details). Dendritic-resolution writing is a prerequisite for Level 5-6 memories (§6) where sensory detail and feature binding matter.

### 2.9 Neural Manifold Constraints

Neural population activity does not occupy the full high-dimensional space of possible firing patterns. Instead, it is confined to **low-dimensional manifolds** — surfaces defined by the intrinsic connectivity of the network ([Gallego et al., 2017](#9-references)). These manifolds are stable over months even as individual neuron tuning changes ([Gallego et al., 2020](#9-references)).

Critically, Sadtler et al. (2014) demonstrated in a closed-loop BCI paradigm that when subjects are required to produce neural activity patterns **within** their intrinsic manifold, learning is rapid. When required to produce patterns **outside** the manifold, learning is severely impaired within a single session ([Sadtler et al., 2014](#9-references)). The network's recurrent connectivity acts as a filter: off-manifold perturbations are rapidly corrected back toward the manifold by recurrent dynamics.

**Design implication:** This is a fundamental constraint on memory writing that the field has not adequately addressed. Any injected pattern must lie **on the intrinsic manifold** of the target neural population, or it will be rejected by recurrent dynamics within milliseconds. The pattern generator must learn the participant's hippocampal and cortical manifolds during calibration and constrain all generated patterns to this subspace. A pattern that is semantically correct but geometrically off-manifold is as useless as random noise. This also means memory writing capacity is bounded by the dimensionality of the intrinsic manifold — you cannot write more distinct memories than the manifold has degrees of freedom to represent.

### 2.10 Phase Precession & Temporal Coding

Hippocampal place cells do not merely fire at higher rates in their preferred locations — they systematically shift their spike timing relative to the ongoing theta oscillation as the animal traverses the place field ([O'Keefe & Recce, 1993](#9-references)). This **phase precession** means that the phase of a spike relative to theta encodes the animal's position within the field with higher resolution than rate coding alone.

At the population level, phase precession compresses behavioral sequences (spanning seconds) into single theta cycles (~125 ms), bringing representations of temporally distant events into the ~20 ms window required for spike-timing-dependent plasticity (STDP) ([Skaggs et al., 1996](#9-references)). This theta-cycle compression is how the hippocampus binds sequential experiences into associative chains.

**Design implication:** Writing episodic or sequential memories requires controlling not just **which** neurons fire but **when within the theta cycle** they fire. A rate-coded injection (correct neurons, wrong timing) will produce a snapshot memory — a static scene without temporal structure. A phase-coded injection (correct neurons at correct theta phases) will produce a sequence memory — an experience with narrative flow. The injection controller must deliver stimulation pulses with theta-phase precision (~5-10 ms resolution within the ~125 ms theta cycle), with earlier phases encoding earlier events in the sequence. This doubles the information bandwidth of injection: rate encodes content identity, phase encodes temporal order.

---

## 3. System Architecture

The system operates in two modes — **de novo writing** and **reconsolidation editing** — that share infrastructure but differ in their entry point and biological mechanism:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MEMORY WRITING BCI                                 │
│                                                                             │
│  MODE A: DE NOVO WRITING                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ CONTENT  │──▶│ PATTERN  │──▶│ PATTERN  │──▶│INTEGRATE │──▶│ VERIFY & │   │
│  │   SPEC   │   │GENERATOR │   │ INJECTOR │   │& CONSOL. │   │ VALIDATE │   │
│  │          │   │          │   │          │   │          │   │          │   │
│  │ Define   │   │ Manifold-│   │ Phase-   │   │ Schema-  │   │ Probe &  │   │
│  │ content, │   │ constrain│   │ precise  │   │ aware    │   │ decode   │   │
│  │ links,   │   │ & engram │   │ delivery │   │ replay   │   │ content  │   │
│  │ scaffold │   │ allocate │   │ + inhib. │   │ engine   │   │ fidelity │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                             │
│  MODE B: RECONSOLIDATION EDITING                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ TARGET   │──▶│REACTIVATE│──▶│DESTABILIZE──▶│ INJECT   │──▶│ RE-      │   │
│  │ MEMORY   │   │ EXISTING │   │& MODIFY  │   │ MODIFIED │   │CONSOLI-  │   │
│  │ SELECT   │   │  TRACE   │   │          │   │ CONTENT  │   │  DATE    │   │
│  │          │   │          │   │          │   │          │   │          │   │
│  │ Identify │   │ Cue to   │   │ Block    │   │ Write    │   │ Allow    │   │
│  │ memory   │   │ make     │   │ protein  │   │ updated  │   │ restabi- │   │
│  │ to edit  │   │ labile   │   │ synth.   │   │ pattern  │   │ lization │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                             │
│  SHARED INFRASTRUCTURE                                                      │
│  ┌────────────────────┐  ┌───────────────────┐  ┌──────────────────────┐    │
│  │  MANIFOLD MODEL    │  │  MEMORY REGISTRY  │  │  SCHEMA LIBRARY      │    │
│  │  (Population       │  │ (Written memories,│  │  (Known cortical     │    │
│  │   geometry)        │  │  status, fidelity)│  │   schemas for fast   │    │
│  │                    │  │                   │  │   consolidation)     │    │
│  └────────────────────┘  └───────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Content Specification

The content specification module translates human-readable memory descriptions into machine-processable formats.

| Component | Function |
|-----------|----------|
| Memory descriptor | Structured specification of memory content: semantic content, sensory features, emotional valence, temporal sequence (with phase-coding annotations for sequential memories). Natural input modalities including inner speech ([Kunz et al., 2025](References.md)) could serve as a high-bandwidth content specification channel — the participant describes the memory they want written, and motor-cortical speech representations are decoded into the descriptor format |
| Associative link map | Defines which existing memories the new memory should connect to (retrieval routes) |
| Spatial/relational scaffold | Grid cell manifold coordinates — position on the toroidal scaffold encoding both spatial location and conceptual topology |
| Schema compatibility checker | Assesses whether content maps onto an existing cortical schema (routes to fast or slow consolidation pathway) |
| Complexity class | Assigns the memory to a complexity level (see §6) which determines the writing protocol |

### 3.2 Neural Pattern Generation

The pattern generator translates content specifications into participant-specific neural activation patterns, subject to manifold constraints.

| Component | Function |
|-----------|----------|
| Manifold-constrained encoder | Generative model (VAE/diffusion) that outputs patterns guaranteed to lie on the participant's intrinsic neural manifold; rejects off-manifold solutions. The architecture draws on latent diffusion approaches that have demonstrated bidirectional mapping between semantic content and neural representations ([Takagi & Nishimoto, 2023](References.md); [Lu et al., 2023](References.md)) — here applied in the generative (content→neural) direction rather than the reconstructive (neural→content) direction |
| Engram allocator | Identifies write-eligible neurons: high-CREB mature neurons + critical-period newborn DGCs; avoids neurons committed to existing engrams |
| Partial-pattern optimizer | Computes minimal seed pattern (~20-30% of engram) sufficient for CA3 autoassociative completion; verifies orthogonality to existing attractors |
| Phase-sequence encoder | For sequential memories, assigns theta-phase timing to each element; earlier events → earlier theta phases |
| Inhibitory context generator | Specifies the required VIP+/SST+/PV+ interneuron activation pattern to create the appropriate disinhibitory window and constrain engram size |
| Index synthesizer | Generates compressed hippocampal index code (~100-1000 dims) encoding the content pointers |

The encoder leverages the BRAND platform for real-time neural computation ([Ali et al., 2024](References.md)), running the generative model with manifold projection as a hard constraint.

### 3.3 Pattern Injection

| Component | Function |
|-----------|----------|
| Multi-site stimulator | Delivers patterned microstimulation to target neuron populations across hippocampus and entorhinal cortex |
| Phase-precise timer | Locks injection to specific theta phases: content identity at theta trough (maximal plasticity), sequence position encoded by phase offset within cycle |
| Inhibitory co-stimulator | Activates VIP+ interneurons to open disinhibitory window at injection site; allows SST+ lateral inhibition to regulate engram boundaries |
| Dosage regulator | Controls stimulation intensity based on real-time neural response; prevents over-excitation |
| State monitor | Verifies brain state is receptive (awake-quiet for initial injection; NREM for replay consolidation) |
| Manifold tracker | Real-time verification that evoked population activity remains on the intrinsic manifold; aborts if activity is pushed off-manifold |

### 3.4 Integration & Consolidation

| Component | Function |
|-----------|----------|
| Artificial replay generator | Creates synthetic ~100ms SWR-like sequences with controlled forward/reverse ordering and phase structure |
| Schema router | Directs schema-compatible memories to fast consolidation pathway; schema-absent memories to standard multi-night replay |
| Sleep-stage coordinator | Times replay delivery to NREM SO-spindle-ripple coupling windows |
| Cortical distribution monitor | Tracks emergence of cortical traces via high-gamma reinstatement patterns |
| Interference detector | Monitors for degradation of existing memories during consolidation |

### 3.5 Reconsolidation Editor (Mode B)

| Component | Function |
|-----------|----------|
| Memory reactivator | Delivers cues that reactivate the target memory, opening the reconsolidation window (~5-6 hours) |
| Lability detector | Confirms the trace has destabilized (signatures: AMPA receptor internalization, protein synthesis dependence) |
| Content modifier | Injects updated pattern elements during the labile phase, overwriting specific engram components |
| Restabilization monitor | Tracks protein synthesis-dependent restabilization; verifies modified content is incorporated |

This mode exploits the reconsolidation mechanism ([Nader et al., 2000](#9-references); [Monfils et al., 2009](#9-references); [Schiller et al., 2010](#9-references)): retrieved memories become transiently labile and protein synthesis-dependent, creating a window during which the trace can be modified before restabilization.

### 3.6 Verification & Validation

| Component | Function |
|-----------|----------|
| Retrieval probe | Presents cues designed to trigger recall of the written memory |
| Content decoder | Decodes neural activity during retrieval to verify content matches specification |
| Fidelity scorer | Quantifies match between intended and actual memory content (see §5.3) |
| Manifold consistency checker | Verifies the written trace occupies a stable point on the neural manifold (not a transient off-manifold perturbation that will decay) |
| Side-effect detector | Checks for unintended changes to existing memories or neural function |

The verification system uses adaptive decoding with memristor-based co-evolutionary learning ([Liu et al., 2025](References.md)) for real-time content assessment. The decoder architecture leverages advances in neural-to-content reconstruction — latent diffusion models can reconstruct high-resolution perceptual content from brain activity ([Takagi & Nishimoto, 2023](References.md)), and controlled reconstruction methods can decode both semantic and structural features ([Lu et al., 2023](References.md)). These approaches, originally developed for fMRI, provide the architectural template for the electrophysiological content decoder used in verification.

---

## 4. Hardware Requirements

Artificial memory writing is **Tier 1 (invasive) only**. Non-invasive approaches (EEG, tACS, tTIS) lack the spatial resolution to write specific neural patterns to identified neuron populations. Tier 2 supports enhancement because it only needs to modulate ongoing activity; writing requires specifying and delivering precise activation patterns.

### 4.1 Write-Capable Electrode Arrays

| Specification | Memory Writing | Memory Enhancement |
|--------------|----------------|-------------------------------|
| Channel count | 1000+ per hemisphere | 100-256 (Tier 1) |
| Electrode pitch | <50 μm (somatic); <10 μm (dendritic, future) | ~400 μm (Utah array) |
| Coverage | Bilateral hippocampus (CA1, CA3, DG) + entorhinal cortex + prefrontal cortex | Unilateral hippocampus + prefrontal cortex |
| Stimulation capability | Patterned multi-site microstimulation (independent current per channel) + phase-precise timing (<5 ms jitter) | Single-site or paired-site stimulation |
| Read/write ratio | Simultaneous read-write on adjacent channels | Primarily read; occasional write |
| Interneuron access | Cell-type-specific channels for VIP+/SST+ co-stimulation | Not required |

Hardware platforms are converging on the specifications required for memory writing. The wireless subdural array by Jung et al. (2025) achieves 65,536 electrodes with 1,024 simultaneous recording channels in a fully wireless, subdural-contained form factor ([Jung et al., 2025](References.md)) — approaching the channel count needed for bilateral hippocampal coverage. Hettick et al. (2025) demonstrate minimally invasive implantation of scalable high-density cortical microelectrode arrays capable of both multimodal neural decoding and stimulation ([Hettick et al., 2025](References.md)), directly addressing the simultaneous read-write requirement. Flexible substrate technologies ([Tang et al., 2023](References.md)) improve chronic biocompatibility and conformability to hippocampal geometry, and wireless power/data architectures ([Won et al., 2023](References.md)) eliminate transcutaneous connectors. The key remaining constraint is achieving read-write interleaving at the required spatial density within deep structures — current high-density arrays target cortical surface, not hippocampal depth.

### 4.2 Optogenetic Delivery

Cell-type-specific activation provides precision that electrical stimulation cannot achieve. Optogenetic approaches use viral vectors (AAV) to express light-sensitive opsins (e.g., ChR2, ChRmine) in target neuron populations. For memory writing, cell-type specificity is not a luxury — it is essential for independently controlling excitatory engram neurons and inhibitory interneuron subtypes.

| Approach | Precision | Speed | Cell-Type Specificity | Suitability |
|----------|-----------|-------|-----------------------|-------------|
| Electrical microstimulation | ~50-100 μm radius | <1 ms | None (activates all nearby cells) | Broad-field activation; cannot selectively target interneuron subtypes |
| Optogenetic (ChR2/AAV) | Single-cell-type | ~1-5 ms | Yes (promoter-dependent) | Separate opsins for excitatory (CaMKII) and inhibitory (PV-Cre, VIP-Cre) populations |
| Two-photon holographic optogenetics | Single-cell (~1 μm) | ~5-10 ms | Yes | Highest precision; dendritic-scale targeting possible; limited to superficial tissue |
| Combined electrical + multi-opsin | Cell-type writing + broad-field read | <1-5 ms | Yes (multi-opsin) | Optimal for memory writing: excitatory engram + inhibitory context + electrical readout |

### 4.3 Processing Requirements

| Requirement | Memory Writing | Memory Enhancement |
|-------------|----------------|-------------------------------|
| Decoder type | Generative (content → manifold-constrained neural pattern) | Discriminative (neural pattern → label) |
| Latency budget | <10 ms (injection must lock to specific theta phase) | <25 ms (closed-loop stimulation) |
| Compute | On-implant neuromorphic + external GPU cluster; neuromorphic architectures designed for brain rewiring applications ([Chiappalone et al., 2022](References.md)) provide the on-implant substrate | On-implant neuromorphic chip |
| Model complexity | Manifold-constrained VAE/diffusion generative model + attractor landscape simulator | CNN/LFADS classifier |
| Calibration data | Paired (content, neural) recordings over weeks + manifold estimation from spontaneous activity | Neural recordings during task performance |
| Storage | Full memory registry + pattern library + manifold model + schema library | Memory index library |

---

## 5. Software Pipeline

### 5.1 Content-to-Neural Encoder

The encoder translates content specifications into neural activation patterns via a five-stage pipeline:

| Stage | Input | Output | Method |
|-------|-------|--------|--------|
| Content embedding | Memory descriptor (text, sensory features, associations, temporal sequence) | High-dimensional content vector (2048-dim) | Multimodal transformer encoder; architecture parallels the latent space used in neural reconstruction models ([Takagi & Nishimoto, 2023](References.md)) but operates in the inverse direction |
| Manifold projection | Content vector + participant manifold model | Target pattern constrained to intrinsic neural manifold | Manifold-constrained latent diffusion model; the diffusion process is conditioned on the content vector and constrained to the participant's neural manifold at each denoising step; rejection sampling for off-manifold proposals |
| Attractor verification | Manifold-constrained pattern + existing attractor landscape | Verified pattern with confirmed unique basin of attraction | Simulate CA3 recurrent dynamics; verify convergence to novel attractor, not blend with existing |
| Partial pattern extraction | Verified full engram pattern | Seed pattern (20-30% of neurons) with phase-timing annotations | CA3 attractor basin analysis; select minimal activating set; assign theta-phase positions |
| Index synthesis | Full engram pattern + associative link map | Compressed hippocampal index (~100-1000 dims) | Learned index compression matching participant's indexing statistics |

The manifold-constrained VAE is trained on two data sources: (1) paired (content, neural) recordings during natural encoding of known content (weeks of calibration), and (2) spontaneous activity recordings for manifold estimation (recorded during rest/sleep). The manifold model is updated continuously as the participant's neural geometry evolves.

### 5.2 Injection Controller

| Stage | Latency Budget | Method |
|-------|---------------|--------|
| Theta phase estimation | <2 ms | Real-time Hilbert transform on hippocampal LFP |
| Phase-sequence scheduling | <1 ms | Map temporal sequence elements to theta-phase positions; precompute per-cycle delivery schedule |
| Pattern loading | <1 ms | Pre-computed patterns buffered in on-implant memory |
| Inhibitory pre-activation | <2 ms | VIP+ opsin pulse to open disinhibitory window ~10 ms before excitatory injection |
| Stimulation delivery | <3 ms | Parallel current sources + optogenetic pulses; one per target channel |
| Manifold tracking | <1 ms | Project evoked population vector onto manifold; compute residual |
| Adaptation | <2 ms | Online adjustment of stimulation amplitude based on evoked response and manifold adherence; adaptive neuromodulation principles ([Lampert et al., 2025](References.md)) applied at the single-injection timescale |

**Total injection latency target:** <10 ms (must complete within a single theta-phase window, delivering sequence elements across the ~125 ms theta cycle with ~5-10 ms phase precision)

### 5.3 Verification Decoder

| Metric | Method | Threshold |
|--------|--------|-----------|
| Pattern similarity | Cosine similarity between target and evoked population vectors | >0.85 for initial injection; >0.90 after consolidation |
| Manifold residual | Distance of evoked pattern from nearest point on intrinsic manifold | <0.1 normalized manifold units (on-manifold) |
| Retrieval latency | Time from cue onset to hippocampal pattern completion | <500 ms (comparable to natural recall) |
| Content fidelity | Decoded content vs. specification using cross-validated decoder | >0.80 semantic similarity score |
| Sequence integrity | Phase-decoded temporal order vs. specified sequence | Kendall tau >0.90 for sequential memories |
| Interference check | Probe retrieval of 10 existing memories pre/post writing | <0.05 fidelity degradation on any existing memory |
| Subjective report | Participant describes the memory; scored against specification | Qualitative match on key content elements |

### 5.4 Artificial Replay Engine

The replay engine generates synthetic sharp-wave ripple sequences containing the written memory content, timed to the SO-spindle-ripple coupling window during NREM sleep.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Replay duration | ~80-120 ms | Matches natural SWR duration ([Davidson et al., 2009](Memory.md)) |
| Replay rate | 1-3 per minute during target NREM periods | Conservative; natural rate is ~1-3 SWR/min |
| Timing lock | SO up-state → spindle trough → ripple | Matches natural SO-spindle-ripple coupling |
| Sequence direction | Forward (60%) + reverse (40%) for episodic; reverse-dominant (80%) for reward-associated | Forward preserves narrative; reverse supports credit assignment ([Foster & Wilson, 2006](Memory.md)) |
| Content | Compressed written memory content in SWR-band temporal code with theta-phase structure preserved | Derived from injection pattern via temporal compression |
| Schema-primed sessions | 1 concentrated session (2-3 hours) for schema-compatible memories | Tse et al. 2007/2011: schema-matching content consolidates in hours |
| Schema-absent sessions | Minimum 3 sleep cycles over 1-3 nights | Standard systems consolidation timeline |
| Interference avoidance | Suppress artificial replay when natural replay of other content is detected | Uses content decoder |

---

## 6. Memory Complexity Hierarchy

Not all artificial memories are equally difficult to write. The following hierarchy orders memory types from simplest to most complex, grounded in the specific circuit-level challenges at each level:

| Level | Memory Type | Example | Write Target | Key Circuit Challenge | Estimated Feasibility |
|-------|-------------|---------|-------------|----------------------|----------------------|
| 1 | Simple association | "X paired with Y" | CA3 Hebbian link between two existing representations | Minimal: strengthen single synapse population; no new engram needed | Near-term (demonstrated in rodents) |
| 2 | Contextual association | "X happened in place Y" | Place cell + content cell co-activation in CA1 | Requires coordinating place and content cells; phase timing matters for binding | Near-term (variant of Ramirez et al. 2013) |
| 3 | Semantic fact | "The capital of France is Paris" | Hippocampal index pointing to cortical semantic representations | Requires content-to-neural encoder; index must activate correct cortical targets via consolidation | Medium-term |
| 4 | Procedural skill | Motor sequence for a new task | Hippocampal-striatal-cerebellar circuit coordination | Multi-region writing with different plasticity rules per region; requires dendritic-branch precision in motor cortex ([Cichon & Bhatt, 2015](#9-references)) | Medium-term |
| 5 | Structured episode | Meeting with specific people, topics, outcomes | Multi-feature hippocampal index + theta-phase sequence coding | Phase precession structure required; multiple sensory modalities bound via dendritic clustering; inhibitory context must be precisely sculpted | Long-term |
| 6 | Rich episodic memory | Full sensory experience with emotion, context, narrative | Distributed hippocampal-cortical trace with amygdala involvement | Full manifold-constrained multi-region writing with dendritic resolution; emotional valence requires amygdala-hippocampal coordination; no current technology sufficient | Speculative |

```
                        ▲
                       ╱ ╲
                      ╱ L6 ╲        Rich episodic — multi-region manifold + dendritic
                     ╱───────╲         resolution + amygdala coordination
                    ╱   L5    ╲     Structured episode — phase sequence coding +
                   ╱───────────╲       multi-feature dendritic clustering
                  ╱     L4      ╲   Procedural — cross-system plasticity rules +
                 ╱───────────────╲     branch-specific motor cortex writing
                ╱       L3        ╲  Semantic fact — content-to-neural encoder +
               ╱───────────────────╲    schema-accelerated consolidation
              ╱         L2          ╲ Contextual association — place-content
             ╱───────────────────────╲   co-activation with theta-phase binding
            ╱           L1            ╲ Simple association — Hebbian link
           ╱─────────────────────────────╲   strengthening between existing reps
          ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
          Regions   Timing     Dendrites   Manifold dims →
```

The key insight is that each level adds a qualitatively different challenge, not merely a quantitative one. L1→L2 adds spatial binding. L2→L3 adds the encoding problem (content has no natural neural representation). L3→L4 adds multi-system coordination with incompatible plasticity rules. L4→L5 adds temporal structure (phase coding). L5→L6 adds emotional valence and the full manifold constraint across distributed cortex.

---

## 7. Novel Design Elements

### 7.1 Partial-Pattern Writing via CA3 Completion

Rather than writing every neuron of a target engram, the system writes only a 20-30% seed pattern and relies on CA3 autoassociative dynamics to complete the remainder (§2.2). This is the core efficiency insight of the architecture.

**Protocol:**
1. Generate full target engram pattern (pattern generator §3.2)
2. Verify the full pattern occupies a unique basin of attraction in CA3 (attractor verification, §5.1)
3. Compute minimal seed: the smallest subset that reliably converges to the full attractor in simulation
4. Inject seed pattern via patterned microstimulation with VIP+ disinhibitory pre-pulse
5. Monitor CA3 population dynamics for attractor convergence (within ~200 ms)
6. If convergence to wrong attractor: abort, adjust seed (increase size or reposition), retry

**Risk:** Wrong attractor convergence creates a blend or substitution memory. The probability of misconvergence increases with memory load — a hippocampus storing many memories has more crowded attractor basins. The system must maintain a running estimate of attractor density and refuse writes when basin spacing falls below a safety threshold.

### 7.2 Index-First Writing Strategy

Instead of writing a complete memory trace, write only the hippocampal index code and let natural consolidation build the cortical representation.

**Advantage:** Reduces the write problem to a single region (hippocampus) and leverages the brain's own consolidation machinery for cortical trace formation.

**Critical refinement — schema routing:** The consolidation timeline depends entirely on whether the written content matches an existing cortical schema. Schema-compatible writes (e.g., a new fact within a well-learned domain) can consolidate in hours via immediate neocortical gene activation ([Tse et al., 2011](#9-references)). Schema-absent writes (e.g., entirely novel content domain) require the standard multi-night replay pathway. The content specification module must assess schema compatibility and route accordingly.

**Limitation:** The cortical trace will be shaped by the participant's existing representations. Written memories may drift from specification as consolidation adapts them to prior knowledge. For schema-primed writes, this drift is a feature (rapid integration) but also means the written memory may be subtly transformed by the schema. Verification (§3.6) must track this drift across consolidation sessions.

### 7.3 Reconsolidation-Based Memory Editing

For many clinical applications (PTSD treatment, memory correction, phobia reduction), the goal is not to write a new memory from scratch but to **modify an existing one**. Reconsolidation editing exploits the finding that retrieved memories become transiently labile ([Nader et al., 2000](#9-references)), creating a window during which the trace can be updated.

**Protocol:**
1. Present cue to reactivate target memory; verify reactivation via hippocampal pattern decoding
2. Confirm lability onset (reconsolidation window: ~10 min to ~6 hours post-reactivation; [Monfils et al., 2009](#9-references))
3. Inject modified content pattern during the labile phase — updated elements only, preserving the existing trace scaffold
4. Allow restabilization (protein synthesis-dependent, ~6 hours)
5. Verify modified content via retrieval probe at 24 hours and 1 week

**Advantage over de novo writing:** Reconsolidation editing does not require engram allocation, manifold-novel patterns, or full consolidation — it modifies an existing trace in place. The modified memory inherits the original's associative connections, spatial context, and consolidation status. This makes it far more practical for near-term clinical applications than de novo writing.

**Key constraint:** Not all memories reconsolidate upon retrieval. Strong, well-consolidated memories and memories retrieved in routine contexts may resist destabilization. The system must confirm lability before proceeding, and abort if the trace remains stable.

### 7.4 Manifold-Constrained Injection

Every injected pattern must lie on the intrinsic manifold of the target neural population (§2.9). This is not merely a quality check — it is a hard physical constraint.

**Implementation:**
1. During calibration, estimate the neural manifold from spontaneous activity using dimensionality reduction (GPFA, LFADS, or autoencoder)
2. The generative model is trained with a manifold-consistency loss: proposed patterns are penalized for off-manifold residual
3. During injection, the manifold tracker projects the real-time evoked population vector onto the manifold and computes the residual
4. If the residual exceeds threshold, the system immediately adjusts stimulation parameters to bring activity back on-manifold

**Implication for capacity:** The dimensionality of the hippocampal manifold (estimated at ~20-50 independent dimensions in rodent CA1) sets a theoretical upper bound on the number of distinct writable memories. Memory capacity scales exponentially with manifold dimensionality but is further limited by attractor basin spacing. The system should estimate remaining capacity before each write and warn when approaching saturation.

### 7.5 Inhibitory Landscape Engineering

Memory writing requires sculpting the inhibitory surround, not just activating excitatory targets (§2.7).

**Three-phase inhibitory protocol:**
1. **Open the gate** — Activate VIP+ interneurons at the injection site ~10 ms before excitatory stimulation. VIP+ cells inhibit SST+ and PV+ interneurons, creating a transient disinhibitory window that permits plasticity in target pyramidal cells.
2. **Write the pattern** — Deliver excitatory stimulation during the disinhibitory window. Active pyramidal cells recruit local SST+ interneurons via lateral excitation.
3. **Close the gate** — Allow SST+ lateral inhibition to suppress surrounding non-target neurons, naturally constraining engram size to ~2-5% of the local population (matching natural engram sparsity; [Stefanelli et al., 2016](#9-references)).

Without inhibitory engineering, two failure modes emerge: (1) engrams that are too large (inadequate SST+ suppression), which degrade pattern separation and cause interference with existing memories, or (2) engrams that fail to stabilize (inadequate disinhibitory gating), which decay within hours.

### 7.6 Phase-Sequence Writing for Episodic Memory

Episodic memories are not static snapshots — they are temporal sequences. Writing them requires encoding temporal order via theta-phase precession (§2.10).

**Protocol:**
1. Decompose the episodic memory specification into an ordered sequence of scene elements (S₁, S₂, ... Sₙ)
2. Assign each element to a theta-phase position: S₁ at late theta phase (360°), Sₙ at early theta phase (~0°), intermediate elements linearly interpolated
3. During injection, deliver stimulation for each element at its assigned phase within a single theta cycle (~125 ms)
4. Repeat across multiple theta cycles to build STDP-mediated associative links between sequential elements
5. During artificial replay, preserve the phase structure in the compressed SWR-format sequence

**Capacity:** A single theta cycle can encode ~5-7 sequential elements (limited by the ~20 ms inter-element spacing required for STDP). Longer sequences require chaining across multiple theta cycles, with overlap at boundaries. This matches the known capacity limit of hippocampal sequence coding and explains why episodic memories have a natural "chunk" structure.

### 7.7 Associative Bridging

A written memory without connections to existing knowledge is an island — hard to retrieve and likely to be forgotten. Associative bridging creates retrieval routes by co-activating existing engrams during the consolidation phase.

**Protocol:**
1. Identify 3-5 existing memories that should associate with the new memory (specified in the associative link map, §3.1)
2. During artificial replay of the written memory, follow each replay with brief reactivation of the linked existing engrams
3. The temporal proximity (~200-500 ms) engages Hebbian association, creating synaptic links between the new and existing traces
4. Over multiple replay cycles, these links strengthen into stable retrieval routes

This mimics the natural process by which new experiences become integrated with prior knowledge through interleaved replay. For schema-compatible memories, bridging accelerates integration; for schema-absent memories, it is essential for preventing orphaned traces.

### 7.8 Memory Authentication Watermarks

Artificial memories should be distinguishable from natural ones — both for the participant's self-knowledge and for forensic/legal purposes. The system embeds neural "watermarks" during writing.

**Approach:** During injection, a subtle secondary pattern is co-activated alongside the memory content. This pattern is below the threshold for conscious awareness but detectable by the BCI during readout. The watermark encodes: (1) timestamp of writing, (2) content specification hash, (3) writing system version, (4) de novo vs. reconsolidation mode.

**Purpose:** Prevents artificial memories from being confused with natural experiences; enables participants to query "was this memory written?"; provides accountability in forensic and legal contexts where memory authenticity matters.

---

## 8. Risk Analysis & Ethics

### 8.1 Safety Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Incorrect content written | **Critical** | Multi-stage verification (§3.6); content decoded after injection, after first consolidation, and after 1 week; abort and erase protocol if fidelity <0.80 |
| Off-manifold injection causing unstable trace | **High** | Manifold tracker with real-time residual monitoring; automatic abort if residual exceeds threshold; pattern is guaranteed on-manifold before delivery |
| Seizure from patterned multi-site stimulation | **High** | Afterdischarge detection with <5 ms halt; conservative current limits; never exceed 100 μA per channel; epileptologist oversight; inhibitory pre-activation (§7.5) provides seizure-protective effect |
| Overwriting existing memories | **Critical** | Engram allocator avoids committed neurons; attractor basin verification ensures orthogonality; interference check probes 10+ existing memories before and after writing; immediate halt if degradation >0.05 |
| Wrong attractor convergence | **High** | Pre-injection attractor landscape simulation; real-time convergence monitoring; automatic abort if trajectory deviates from target basin; attractor density tracking to refuse writes when basins are crowded |
| Pathological engram size (too large or too small) | **High** | SST+ interneuron-mediated size regulation via inhibitory landscape engineering (§7.5); real-time engram size estimation; abort if outside 1-8% of local population |
| Tissue damage from chronic high-density stimulation | **High** | Biocompatible electrode materials; charge-balanced biphasic pulses; impedance monitoring; stimulation holidays between writing sessions |
| Uncontrolled replay propagation | **Medium** | Monitor replay content for unintended spreading activation; throttle artificial replay rate; sleep-stage gating prevents replay during REM |
| Schema-driven content distortion | **Medium** | Verification at 24h and 1 week to track schema-mediated drift from specification; re-inject corrections via reconsolidation editing if drift exceeds tolerance |
| Reconsolidation editing destabilizes non-target memories | **Medium** | Narrow reactivation cue design targeting only the intended trace; verify non-target memory integrity post-edit |

### 8.2 Ethical Considerations

| Issue | Consideration |
|-------|--------------|
| Memory authenticity & identity | Written memories may feel indistinguishable from lived experience. This challenges the participant's sense of self and the authenticity of their autobiographical narrative. Watermarking (§7.8) partially addresses this but cannot prevent subjective confusion. The philosophical question — are you the sum of your experiences if some were never experienced? — has no technical answer. |
| Consent complexity | A participant cannot give informed consent about a memory they haven't yet experienced. The written memory may change their preferences, beliefs, or personality in ways they cannot anticipate. Reconsolidation editing is particularly fraught: modifying an existing memory alters the person's relationship to their own past. Staged consent protocols required. |
| Coercive implantation | Military, employer, or state-mandated memory writing could constitute a profound violation of cognitive liberty. Strict regulatory frameworks must prohibit non-voluntary memory writing. Reconsolidation editing raises the additional risk of covert memory modification — altering someone's memories without their knowledge. |
| False confessions & forensic risk | Written memories could be used to implant false confessions or fabricated eyewitness testimony. Memory authentication watermarks (§7.8) and legal frameworks must address this. Courts must develop standards for distinguishing natural from artificial memories. |
| Irreversibility | Unlike data on a hard drive, a consolidated memory cannot be cleanly deleted. Reconsolidation-based disruption ([Nader et al., 2000](#9-references)) may partially address this, but complete erasure of a written memory is not guaranteed — especially after schema integration has distributed the trace across cortex. |
| Unequal access | Memory writing is Tier 1 only (surgical). This creates extreme inequality between those with access to cognitive augmentation and those without. |
| Therapeutic vs. enhancement boundary | Writing memories for an amnesic patient is therapeutic. Writing them for a healthy student is enhancement. The line between these is blurry and must be drawn through clinical and regulatory consensus. |
| Informed consent paradox | The memory itself may alter the participant's capacity or willingness to consent to future writing. Recursive consent frameworks needed. |

### 8.3 Open Technical Questions

| Question | Current Status | Path Forward |
|----------|---------------|-------------|
| Hippocampal manifold dimensionality in humans | Estimated ~20-50 dims in rodent CA1; unknown in human | Map manifold structure in epilepsy monitoring patients with high-density recordings |
| Manifold stability under repeated writing | Manifolds stable over months for natural activity ([Gallego et al., 2020](#9-references)); unknown for artificial perturbation | Longitudinal tracking in animal models with repeated optogenetic engram creation |
| Engram allocation precision | CREB manipulation demonstrated in rodents; no human equivalent | Develop excitability profiling biomarkers; calcium imaging in surgical patients |
| Newborn neuron targeting in humans | Adult DG neurogenesis confirmed but extent debated | Develop biomarkers for critical-period newborn neurons accessible to implanted electrodes |
| Replay cycles needed for consolidation | Unknown for artificial content; 50-200 natural replays per night | Parametric studies in animal models with optogenetic replay injection |
| Schema compatibility assessment | No automated method; requires domain knowledge | Train schema-matching models on participant's semantic knowledge base |
| Dendritic-resolution writing feasibility | Two-photon holographic optogenetics achieves ~1 μm in superficial cortex; not in deep hippocampus; scalable high-density arrays now achieving electrode pitches compatible with dendritic-scale targeting in cortex ([Hettick et al., 2025](References.md)) | Develop deep-tissue holographic approaches; microLED arrays on electrode shanks; leverage minimally invasive high-density arrays for cortical dendritic writing as a stepping stone |
| Reconsolidation boundary conditions | Not all memories reconsolidate upon retrieval; conditions unclear | Systematic parametric studies varying memory age, strength, retrieval context |
| Minimum viable partial pattern | ~20-30% estimated from CA3 connectivity studies | Empirical testing in rodent CA3 with optogenetic partial-pattern activation |
| Selective erasure of failed writes | Reconsolidation disruption partially effective | Targeted reconsolidation-based erasure protocols; combine with optogenetic inactivation |
| Phase-sequence fidelity | Phase precession well-characterized for spatial coding; unknown whether it can be artificially induced | Test whether externally driven phase-structured stimulation produces STDP-linked sequences in hippocampal slices |

---

## 9. References

This design draws on papers catalogued in this repository:

- **[References.md](References.md)** — BCI platforms, neural decoding, neuromorphic hardware, wireless systems, memory-BCI interfaces
- **[Memory.md](Memory.md)** — Hippocampal circuits, engrams, replay dynamics, grid cells, spatial memory, synaptic plasticity, oscillations

### Key papers by topic

| Topic | Key References |
|-------|---------------|
| Engrams & CREB allocation | Josselyn & Tonegawa 2020; Guskjolen & Cembrowski 2023; Han et al. 2007; Ramirez et al. 2013 |
| CA3 autoassociation | Le Duigou et al. 2014; Sammons et al. 2024; Watson et al. 2024 |
| Replay & consolidation | Davidson et al. 2009; Carr et al. 2011; Berners-Lee et al. 2022; Vaz et al. 2020; Foster & Wilson 2006 |
| Spatial scaffolds | Gardner et al. 2022; Chandra et al. 2025 |
| Synaptic plasticity | Kennedy 2013; Nader et al. 2000 |
| Prefrontal memory | Yadav et al. 2022; Simons & Spiers 2003 |
| Inhibitory circuits | Stefanelli et al. 2016; Letzkus et al. 2015; Lovett-Barron et al. 2014 |
| Dendritic computation | Govindarajan et al. 2006; Kastellakis et al. 2015; Cichon & Bhatt 2015 |
| Neural manifolds | Gallego et al. 2017; Gallego et al. 2020; Sadtler et al. 2014 |
| Phase coding | O'Keefe & Recce 1993; Skaggs et al. 1996 |
| Schema consolidation | Tse et al. 2007; Tse et al. 2011 |
| Reconsolidation | Nader et al. 2000; Monfils et al. 2009; Schiller et al. 2010 |
| Neurogenesis & memory | Sahay et al. 2011; Aimone et al. 2011 |
| BCI platforms | Ali et al. 2024 (BRAND); Liu et al. 2025 (memristor decoder) |
| High-density electrode arrays | Jung et al. 2025 (65K-electrode wireless); Hettick et al. 2025 (minimally invasive, multimodal decode+stim) |
| Flexible/wireless implants | Tang et al. 2023; Won et al. 2023 |
| Neural content reconstruction | Takagi & Nishimoto 2023 (latent diffusion); Lu et al. 2023 (MindDiffuser) |
| Neuromorphic neuroprostheses | Chiappalone et al. 2022 |
| Adaptive neuromodulation | Lampert et al. 2025; Johnson et al. 2013 |
| Inner speech decoding | Kunz et al. 2025 |
| Memory BCI | Burke et al. 2015 |
| Cortical oscillations | Mendoza-Halliday et al. 2024; Yaffe et al. 2017 |

### Additional sources informing the design

- **False memory creation:** Ramirez, S., Liu, X., Lin, P.A., et al. (2013). Creating a false memory in the hippocampus. *Science*, 341(6144), 387-391. doi:10.1126/science.1239073
- **CREB neuronal allocation:** Han, J.H., Kushner, S.A., Yiu, A.P., et al. (2007). Neuronal competition and selection during memory formation. *Science*, 316(5823), 457-460. doi:10.1126/science.1139438
- **Reconsolidation:** Nader, K., Schafe, G.E., & Le Doux, J.E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726. doi:10.1038/35021052
- **Reconsolidation update:** Monfils, M.F., Cowansage, K.K., Klann, E., & LeDoux, J.E. (2009). Extinction-reconsolidation boundaries: key to persistent attenuation of fear memories. *Molecular Psychiatry*, 14, 954-958.
- **Human reconsolidation:** Schiller, D., Monfils, M.F., Raio, C.M., et al. (2010). Preventing the return of fear in humans using reconsolidation update mechanisms. *Nature*, 463, 49-53. doi:10.1038/nature08637
- **Inhibitory engram sculpting:** Stefanelli, T., Bertollini, C., Lüscher, C., et al. (2016). Hippocampal somatostatin interneurons control the size of neuronal memory ensembles. *Neuron*, 89(5), 1074-1085.
- **Disinhibition and learning:** Letzkus, J.J., Wolff, S.B.E., & Lüthi, A. (2015). Disinhibition, a circuit mechanism for associative learning and memory. *Neuron*, 88, 264-276.
- **Dendritic inhibition:** Lovett-Barron, M., Kaifosh, P., Kheirbek, M.A., et al. (2014). Dendritic inhibition in the hippocampus supports fear learning. *Science*, 343, 857-863. doi:10.1126/science.1247485
- **Clustered plasticity:** Govindarajan, A., Kelleher, R.J., & Bhatt, S. (2006). A clustered plasticity model of long-term memory engrams. *Nature Reviews Neuroscience*, 7, 575-583.
- **Dendritic clustering:** Kastellakis, G., Cai, D.J., Mednick, S.C., Silva, A.J., & Bhatt, S. (2015). Synaptic clustering within dendrites: an emerging theory of memory formation. *Progress in Neurobiology*, 126, 19-35.
- **Branch-specific plasticity:** Cichon, J. & Bhatt, W.B. (2015). Branch-specific dendritic Ca²⁺ spikes cause persistent synaptic plasticity. *Nature*, 520, 180-185. doi:10.1038/nature14251
- **Neural manifolds:** Gallego, J.A., Perich, M.G., Miller, L.E., & Solla, S.A. (2017). Neural manifolds for the control of movement. *Neuron*, 94(5), 978-984.
- **Manifold stability:** Gallego, J.A., et al. (2020). Long-term stability of cortical population dynamics underlying consistent behavior. *Nature Neuroscience*, 23, 260-270.
- **Neural constraints on learning:** Sadtler, P.T., et al. (2014). Neural constraints on learning. *Nature*, 512, 423-426. doi:10.1038/nature13665
- **Phase precession:** O'Keefe, J. & Recce, M.L. (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. *Hippocampus*, 3, 317-330.
- **Theta compression:** Skaggs, W.E., McNaughton, B.L., Wilson, M.A., & Barnes, C.A. (1996). Theta phase precession in hippocampal neuronal populations and the compression of temporal sequences. *Hippocampus*, 6, 149-172.
- **Schema consolidation:** Tse, D., Langston, R.F., Kakeyama, M., et al. (2007). Schemas and memory consolidation. *Science*, 316(5821), 76-82. doi:10.1126/science.1135935
- **Schema gene activation:** Tse, D., Takeuchi, T., Kakeyama, M., et al. (2011). Schema-dependent gene activation and memory encoding in neocortex. *Science*, 333(6044), 891-895.
- **Neurogenesis and pattern separation:** Sahay, A., Scobie, K.N., Hill, A.S., et al. (2011). Increasing adult hippocampal neurogenesis is sufficient to improve pattern separation. *Nature*, 472, 466-470.
- **Newborn neuron properties:** Aimone, J.B., Deng, W., & Gage, F.H. (2011). Resolving new memories: a critical look at the dentate gyrus, adult neurogenesis, and pattern separation. *Neuron*, 70, 589-596.
- **High-density wireless array:** Jung, T., Zeng, N., Fabbri, J.D., et al. (2025). A wireless subdural-contained brain–computer interface with 65,536 electrodes and 1,024 channels. *Nature Electronics*, 8, 1272-1288. doi:10.1038/s41928-025-01509-9
- **Minimally invasive high-density arrays:** Hettick, M., Ho, E., Poole, A.J., et al. (2025). Minimally invasive implantation of scalable high-density cortical microelectrode arrays for multimodal neural decoding and stimulation. *Nature Biomedical Engineering*. doi:10.1038/s41551-025-01501-w
- **Latent diffusion neural reconstruction:** Takagi, Y. & Nishimoto, S. (2023). High-resolution image reconstruction with latent diffusion models from human brain activity. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 14453-14463.
- **Controlled neural reconstruction:** Lu, Y., Du, C., Zhou, Q., Wang, D., & He, H. (2023). MindDiffuser: Controlled image reconstruction from human brain activity with semantic and structural diffusion. *Proceedings of the 31st ACM International Conference on Multimedia*, 5899-5908. doi:10.1145/3581783.3613832
- **Flexible BCIs:** Tang, X., Shen, H., Zhao, S., et al. (2023). Flexible brain–computer interfaces. *Nature Electronics*, 6, 109-118. doi:10.1038/s41928-022-00913-9
- **Neuromorphic neuroprostheses:** Chiappalone, M., Cota, V.R., Carè, M., et al. (2022). Neuromorphic-based neuroprostheses for brain rewiring: state-of-the-art and perspectives in neuroengineering. *Brain Sciences*, 12(11), 1578.
- **Adaptive neuromodulation:** Lampert, F., Baker, M.R., Jensen, M.A., et al. (2025). Adaptive neuromodulation dialogues: navigating current challenges and emerging innovations in neuromodulation system development. *Journal of Neural Engineering*, 22(6), 061005. doi:10.1088/1741-2552/ae2359
- **Neuromodulation for brain disorders:** Johnson, M.D., Lim, H.H., Netoff, T.I., et al. (2013). Neuromodulation for brain disorders: challenges and opportunities. *IEEE Transactions on Biomedical Engineering*, 60(3), 610-624. doi:10.1109/TBME.2013.2244890
- **Inner speech decoding:** Kunz, E.M., Abramovich Krasa, B., Kamdar, F., et al. (2025). Inner speech in motor cortex and implications for speech neuroprostheses. *Cell*, 188(17), 4658-4673.e17. doi:10.1016/j.cell.2025.06.015
- **MIMO hippocampal prosthesis:** Berger, T.W., Song, D., et al. — multi-input multi-output model replacing damaged hippocampal circuitry; architecture used here in reverse (generative) mode

