# Vietnamese Flood and Storm Event Information Extraction Dataset
## Named Entity Recognition Annotation Guidelines

**Version:** 1.0  
**Task:** Sentence-level Named Entity Recognition (NER)  
**Language:** Vietnamese

---

## 1. Task Definition

The objective of this annotation task is to identify and classify named entities and event-related information associated with floods, storms, and other natural disasters in Vietnamese news articles.

The annotation is performed at the sentence level. Annotators must identify all relevant entity spans explicitly mentioned in each sentence and assign one of the ten predefined entity labels.

The annotation focuses on information directly related to disaster events, including disaster types, weather conditions, affected locations, temporal expressions, damage, casualties, affected populations, response activities, response agencies, and disaster warning or intensity levels.

### 1.1. General Annotation Principles

Annotators must follow these principles:

1. **Shortest Meaningful Span:** Select the shortest span that preserves the complete meaning of the entity.
2. **Event Relevance:** Annotate entities directly related to floods, storms, inundation, landslides, and other relevant disaster events.
3. **Explicit Information:** Do not infer or create entities based on information that is not explicitly stated in the sentence.
4. **Complete Annotation:** Annotate all valid entities, including multiple entities belonging to the same label within one sentence.
5. **Sentence-level Scope:** Annotate entities independently at the sentence level. Event grouping or cross-sentence entity linking is not required.
6. **Text Preservation:** Preserve the original Vietnamese text. Do not normalize spelling, rewrite expressions, or modify entity boundaries to improve readability.
7. **Non-overlapping Spans:** Do not create overlapping or nested entity spans in the primary annotation dataset. When an expression is part of a larger named organization, annotate the complete organization name rather than extracting an overlapping location entity.

---

## 2. Entity Label Definitions

The dataset contains ten entity labels.

### 2.1. DISASTER_TYPE

**Definition:** The type of natural disaster or primary disaster phenomenon explicitly mentioned in the sentence.

**Examples:**
- bão
- lũ
- ngập lụt
- sạt lở
- lũ quét
- triều cường

**Annotation rules:**
- Annotate the shortest complete expression identifying the disaster phenomenon.
- Distinguish the disaster itself from its meteorological causes.
- Annotate multiple disaster types separately when they are explicitly mentioned.

**Example:**

Sentence: "Mưa lớn gây lũ quét và sạt lở đất tại khu vực miền núi."

- `lũ quét` → DISASTER_TYPE
- `sạt lở đất` → DISASTER_TYPE

### 2.2. LOCATION

**Definition:** Specific geographic names and administrative locations directly associated with the disaster event, its impacts, or response activities.

**Examples:**
- Quảng Nam
- huyện A Lưới
- xã Hòa Bắc
- TP Đà Nẵng
- sông Hương

**Annotation rules:**
- Annotate specific geographic names, including provinces, districts, communes, cities, rivers, mountains, and other identifiable geographic entities.
- For consecutive administrative units referring to the same location, annotate the complete contiguous geographic expression.
- Do not annotate generic spatial expressions that do not identify a specific geographic entity.
- Do not annotate general destination or environmental descriptions as locations.

**Not annotated as LOCATION:**
- nơi an toàn
- khu vực nguy hiểm
- vùng trũng thấp
- khu dân cư
- vùng bị ngập

**Example:**

Sentence: "Người dân tại xã Hòa Bắc, huyện Hòa Vang, TP Đà Nẵng được sơ tán đến nơi an toàn."

- `xã Hòa Bắc, huyện Hòa Vang, TP Đà Nẵng` → LOCATION
- `nơi an toàn` → Not annotated

### 2.3. TIME

**Definition:** Temporal expressions indicating when a disaster occurs, is expected to occur, or when a response activity takes place.

**Examples:**
- sáng 15/10
- từ ngày 12 đến 14/10
- trong đêm
- hôm qua
- sáng nay
- cùng ngày

**Annotation rules:**
- Annotate both absolute and relative temporal expressions.
- Include time expressions related to observed events, forecasts, and response activities.
- Preserve the original expression without converting it into a standardized date or timestamp.
- Do not infer missing dates from the publication date.

**Example:**

Sentence: "Bão dự kiến đổ bộ vào miền Trung trong đêm 15/10."

- `trong đêm 15/10` → TIME

### 2.4. WEATHER_CAUSE

**Definition:** Meteorological conditions and weather-related phenomena that cause, contribute to, or intensify disaster events.

**Examples:**
- mưa lớn
- áp thấp nhiệt đới
- hoàn lưu bão
- mưa kéo dài
- gió mạnh
- không khí lạnh

**Annotation rules:**
- Annotate weather-related causes or conditions rather than the resulting disaster.
- Distinguish meteorological causes from disaster types.
- Do not annotate ordinary weather descriptions unless they are relevant to the disaster event.

**Example:**

Sentence: "Mưa lớn kéo dài khiến nhiều khu vực xảy ra lũ quét."

- `Mưa lớn kéo dài` → WEATHER_CAUSE
- `lũ quét` → DISASTER_TYPE

### 2.5. DAMAGE

**Definition:** Physical and material damage caused by disaster events, including damage to buildings, infrastructure, agriculture, transportation systems, and property.

**Examples:**
- 120 nhà bị tốc mái
- 5km đường bị chia cắt
- nhiều công trình bị hư hỏng
- hàng trăm hecta lúa bị ngập
- 20 căn nhà bị sập

**Annotation rules:**
- Annotate the complete meaningful expression describing the damage.
- Include quantities, affected objects, and damage conditions when they form a complete damage expression.
- Do not annotate the number or object alone if the surrounding expression is necessary to represent the damage.
- Do not classify human deaths, missing persons, or injuries as DAMAGE.

**Example:**

Sentence: "Mưa lũ khiến 20 căn nhà bị sập và 5km đường bị hư hỏng."

- `20 căn nhà bị sập` → DAMAGE
- `5km đường bị hư hỏng` → DAMAGE

### 2.6. CASUALTY

**Definition:** Direct human casualties resulting from disaster events, including deaths, missing persons, and injuries.

**Examples:**
- 2 người mất tích
- 5 người bị thương
- 1 người tử vong
- 3 người thiệt mạng
- 4 người bị thương nặng

**Annotation rules:**
- Annotate the complete expression describing human casualties.
- Include quantities and casualty status when available.
- Distinguish casualties from people who are merely affected, evacuated, isolated, or displaced.
- Do not classify property damage as CASUALTY.

**Example:**

Sentence: "Trận lũ khiến 2 người mất tích và 5 người bị thương."

- `2 người mất tích` → CASUALTY
- `5 người bị thương` → CASUALTY

### 2.7. AFFECTED_POPULATION

**Definition:** People, households, or population groups affected by disaster events without necessarily being reported as casualties.

**Examples:**
- hơn 500 hộ dân
- hàng nghìn người
- 200 học sinh
- 300 hộ gia đình
- người dân vùng ngập

**Annotation rules:**
- Annotate people or population groups explicitly identified as affected, evacuated, isolated, displaced, or otherwise impacted by the disaster.
- Include quantities and qualifiers when they are part of the population expression.
- Do not classify deaths, missing persons, or injuries as AFFECTED_POPULATION.
- Do not annotate generic references to people unless their relationship to the disaster is clear from the sentence.

**Example:**

Sentence: "Hơn 500 hộ dân phải sơ tán khỏi khu vực ngập lụt."

- `Hơn 500 hộ dân` → AFFECTED_POPULATION
- `sơ tán` → RESPONSE_ACTION

### 2.8. RESPONSE_ACTION

**Definition:** Actions undertaken by individuals, organizations, authorities, or emergency forces to prepare for, prevent, respond to, or recover from disaster events.

**Examples:**
- tổ chức sơ tán
- di dời người dân
- cấm biển
- khắc phục sạt lở
- gia cố đê điều
- cứu hộ người dân
- triển khai lực lượng ứng cứu

**Annotation rules:**
- Annotate the complete meaningful action phrase.
- Include the main action and necessary verbal modifiers.
- Do not include the object of the action when it can be independently annotated under another label.
- Distinguish response actions from the agencies performing them.

**Example:**

Sentence: "Lực lượng chức năng tổ chức sơ tán 200 hộ dân đến nơi an toàn."

- `Lực lượng chức năng` → RESPONSE_AGENCY
- `tổ chức sơ tán` → RESPONSE_ACTION
- `200 hộ dân` → AFFECTED_POPULATION

### 2.9. RESPONSE_AGENCY

**Definition:** Government agencies, organizations, institutions, and emergency forces responsible for disaster prevention, response, rescue, or recovery activities.

**Examples:**
- UBND tỉnh
- Ban Chỉ huy PCTT
- lực lượng chức năng
- Bộ đội Biên phòng
- lực lượng cứu hộ
- Công an tỉnh Quảng Nam

**Annotation rules:**
- Annotate the complete organization or agency name.
- Include administrative identifiers when they form part of the organization's name.
- Do not extract a LOCATION span from inside an already annotated organization name.
- Generic references such as "lực lượng chức năng" may be annotated when they clearly refer to an entity performing disaster response activities.

**Example:**

Sentence: "UBND tỉnh Quảng Nam huy động lực lượng chức năng hỗ trợ người dân."

- `UBND tỉnh Quảng Nam` → RESPONSE_AGENCY
- `lực lượng chức năng` → RESPONSE_AGENCY

### 2.10. WARNING_LEVEL

**Definition:** Explicit warning levels, disaster risk classifications, alert stages, and quantitative intensity measurements that describe the severity or intensity of disaster events.

This label includes both officially classified warning levels and measurable disaster intensity indicators.

**Examples:**

Official warning and classification levels:
- cấp độ rủi ro cấp 3
- bão cấp 12
- giật cấp 14
- báo động 2
- mức cảnh báo cấp 4

Quantitative disaster intensity:
- ngập 2m
- ngập 3m
- mực nước dâng 1,5m
- độ sâu ngập 2m

**Annotation rules:**
- Annotate explicit warning levels, disaster risk classifications, alert stages, and intensity measurements.
- Include the complete meaningful expression when it represents a warning or intensity level.
- Flood depth measurements are included because they indicate the intensity and severity of inundation.
- Do not infer warning levels or intensity values that are not explicitly stated.
- Distinguish intensity measurements from ordinary weather measurements that do not directly represent disaster severity.
- When multiple intensity levels or measurements are mentioned, annotate each separately.

**Example 1:**

Sentence: "Bão mạnh cấp 12, giật cấp 14."

- `cấp 12` → WARNING_LEVEL
- `cấp 14` → WARNING_LEVEL

**Example 2:**

Sentence: "Nhiều khu vực bị ngập sâu từ 2m đến 3m."

- `ngập sâu từ 2m đến 3m` → WARNING_LEVEL

**Example 3:**

Sentence: "Mực nước tại khu vực ngập đạt độ sâu 2m."

- `độ sâu 2m` → WARNING_LEVEL

**Important distinction:**

Ordinary rainfall measurements, such as "lượng mưa đạt 200mm", should not automatically be annotated as WARNING_LEVEL. They are not necessarily direct indicators of the disaster intensity category defined by this dataset.

---

## 3. Label Disambiguation Rules

The following distinctions must be applied consistently.

| Label Pair | Disambiguation Rule |
|---|---|
| DAMAGE vs. CASUALTY | DAMAGE describes material or infrastructure damage. CASUALTY describes human deaths, missing persons, and injuries. |
| CASUALTY vs. AFFECTED_POPULATION | CASUALTY refers to direct human casualties. AFFECTED_POPULATION refers to affected people or households without necessarily implying injury or death. |
| DISASTER_TYPE vs. WEATHER_CAUSE | DISASTER_TYPE identifies the disaster phenomenon. WEATHER_CAUSE identifies the meteorological conditions contributing to the event. |
| LOCATION vs. RESPONSE_AGENCY | LOCATION identifies specific geographic entities. RESPONSE_AGENCY identifies organizations or forces performing response activities. |
| LOCATION vs. Generic Spatial Expressions | Only specific geographic names are LOCATION. Generic expressions such as "nơi an toàn" and "vùng trũng thấp" are not LOCATION. |
| WEATHER_CAUSE vs. WARNING_LEVEL | WEATHER_CAUSE describes meteorological conditions. WARNING_LEVEL describes explicit warning classifications or disaster intensity indicators. |
| DAMAGE vs. WARNING_LEVEL | DAMAGE describes actual physical impacts. WARNING_LEVEL describes warning or intensity measurements, such as flood depth. |
| RESPONSE_ACTION vs. RESPONSE_AGENCY | RESPONSE_ACTION identifies an activity. RESPONSE_AGENCY identifies the organization or force performing it. |

---

## 4. Special Annotation Cases

### 4.1. Multi-level Geographic Names

When multiple consecutive administrative units refer to the same geographic location, annotate the complete expression as one LOCATION span.

Example:

"Xã Hòa Bắc, huyện Hòa Vang, TP Đà Nẵng"

→ LOCATION

Do not split this expression into separate overlapping entities.

### 4.2. Relative Time Expressions

Annotate relative time expressions exactly as they appear.

Examples:
- hôm qua
- sáng nay
- cùng ngày
- đêm trước
- ngày hôm sau

Do not normalize these expressions during the initial annotation phase.

### 4.3. Ambiguous Quantities

Preserve numerical qualifiers as part of the entity span when they contribute to its meaning.

Examples:
- hơn 500 hộ dân → AFFECTED_POPULATION
- khoảng 20 căn nhà bị sập → DAMAGE
- gần 3.000 người bị ảnh hưởng → AFFECTED_POPULATION

Do not remove approximation terms such as "hơn", "khoảng", or "gần".

### 4.4. Negation and Unconfirmed Information

Annotation follows the explicit linguistic content of the sentence rather than independently verifying whether an event occurred.

For example:

"Hiện chưa ghi nhận người mất tích."

The expression "người mất tích" is still a CASUALTY mention, although the sentence explicitly negates its occurrence.

Similarly, predicted, potential, and conditional disaster events may be annotated when their relevant entity expressions are explicitly present.

Annotators must not interpret a prediction as confirmation that the event has already occurred.

### 4.5. Forecasts and Historical Events

Annotate entities in:
- Historical disaster reports.
- Current disaster reports.
- Disaster forecasts.
- Potential or conditional disaster scenarios.

No separate label is assigned based on the temporal or factual status of the event.

### 4.6. Response Actions and Their Objects

Annotate the action separately from its affected object whenever the object belongs to another entity category.

Example:

"Di dời 200 hộ dân đến nơi an toàn."

- `Di dời` → RESPONSE_ACTION
- `200 hộ dân` → AFFECTED_POPULATION
- `nơi an toàn` → Not annotated

### 4.7. Personal Names

Individual victim names are not included in the annotation schema.

Annotators must not assign any of the ten labels to personal names solely because they identify disaster victims.

For public dataset release, potentially identifying personal information should be reviewed separately.

### 4.8. Multiple Events in One Sentence

A sentence may mention multiple disaster events or locations.

Annotate all relevant entity mentions independently.

Event grouping, event coreference, and cross-sentence relationships are outside the scope of the primary annotation task.

---

## 5. Span Boundary and Annotation Constraints

### 5.1. Span Boundary

- Select the shortest span that retains the complete semantic meaning of the entity.
- Include quantities and qualifiers when they are integral to the entity expression.
- Exclude unrelated surrounding words.
- Exclude leading or trailing punctuation.
- Preserve Vietnamese diacritics and original character sequences.

### 5.2. Non-overlapping Entities

The primary dataset uses non-overlapping entity spans.

When a geographic name is embedded within an organization name, annotate the complete organization name as RESPONSE_AGENCY and do not create an overlapping LOCATION entity.

### 5.3. Repeated Entities

If an entity appears multiple times in a sentence, annotate every occurrence separately.

### 5.4. Unclear or Ambiguous Expressions

If an expression cannot be confidently assigned to any predefined label, do not annotate it.

Do not force an annotation simply to increase entity coverage.

### 5.5. Context Usage

Annotators may consult surrounding sentences to understand references or resolve ambiguity.

However:
- Only text explicitly present in the target sentence may be annotated.
- Do not copy entities from neighboring sentences.
- Do not infer missing geographic names, dates, quantities, or agencies.

---

## 6. Annotation Quality Control

To ensure consistency and reliability, the annotation process follows these procedures.

### 6.1. Independent Annotation

Gold test sentences must be annotated independently by two human annotators.

Annotators must not access each other's annotations before completing their independent work.

### 6.2. Inter-Annotator Agreement

Agreement is evaluated using:
- Exact-span Precision, Recall, and F1.
- Entity-level agreement by label.
- Boundary disagreement analysis.
- Label disagreement analysis.
- Cohen's Kappa where appropriate for the defined annotation unit.

### 6.3. Adjudication

Disagreements are reviewed and resolved using the official annotation guideline.

The adjudicated annotations form the final gold-standard test set.

### 6.4. Technical Validation

All annotation outputs must pass the following checks:

1. Valid entity labels.
2. Correct character offsets.
3. Exact correspondence between annotated spans and original text.
4. No invalid or unintended overlapping spans.
5. No duplicated entity records.
6. Preservation of sentence and document identifiers.
7. Consistent annotation format across all dataset splits.

---

## 7. Annotation Scope Summary

The annotation task covers ten predefined entity categories:

1. DISASTER_TYPE
2. WEATHER_CAUSE
3. LOCATION
4. TIME
5. DAMAGE
6. CASUALTY
7. AFFECTED_POPULATION
8. RESPONSE_ACTION
9. RESPONSE_AGENCY
10. WARNING_LEVEL

The primary objective is to identify explicit disaster-related information at the sentence level while maintaining consistent entity boundaries and label distinctions.

The task does not require event-level grouping, temporal normalization, geographic normalization, event coreference, or causal relation extraction.

All annotators and automated annotation systems must follow the same version of this guideline to ensure annotation consistency and dataset reproducibility.