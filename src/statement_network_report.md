# Statement-Statement Opinion Network Analysis

## 1. Network Construction

A statement-statement opinion network was constructed using the responses of 87 respondents across 60 survey statements. Nine respondents were removed because they answered fewer than 50% of the survey statements, leaving 87 respondents for network construction.

Each node in the network represents one survey statement, identified by its statement ID and its T/E/S/V category. An edge between two statements represents a sufficiently strong association between respondents' answers to those two statements.

Because the survey responses are ordinal, **Spearman correlation** was used to measure the association between statement responses.

Before calculating correlations, the response matrix was rescaled on a per-respondent basis to reduce the effect of response-style differences between respondents. Pairwise correlations were calculated using at least 15 overlapping responses.

An edge was retained when

$$
|r_{ij}| \geq 0.30
$$

where \(r_{ij}\) is the Spearman correlation between statements \(i\) and \(j\).

The threshold of 0.30 was selected for the final network. Lower thresholds produced substantially denser networks:

| Threshold | Edges | Density | Negative Edges |
|---:|---:|---:|---:|
| 0.10 | 1546 | 0.8734 | 31 |
| 0.20 | 1211 | 0.6842 | 5 |
| 0.25 | 1021 | 0.5768 | 3 |
| 0.30 | 810 | 0.4576 | 0 |

The 0.30 threshold provides a more selective network while retaining a large connected structure.

---

# 2. Overall Network Structure

The final network contains:

| Measure | Result |
|---|---:|
| Number of nodes | 60 |
| Number of edges | 810 |
| Density | 0.4576 |
| Average degree | 27.00 |
| Positive edges | 810 |
| Negative edges | 0 |
| Connected components | 4 |
| Largest component | 57 / 60 nodes |
| Isolated statements | E02, E03, E04 |

For 60 nodes, the maximum possible number of undirected edges is

$$
\binom{60}{2}
=
\frac{60(59)}{2}
=
1770.
$$

Therefore, the observed network density is

$$
\text{Density}
=
\frac{810}{1770}
=
0.4576.
$$

Thus, approximately **45.8% of all possible statement pairs** have an absolute Spearman correlation of at least 0.30.

The average degree is calculated as

$$
\text{Average degree}
=
\frac{2E}{N}
=
\frac{2(810)}{60}
=
27.
$$

Therefore, a statement is connected to approximately 27 of the other 59 statements on average.

### Interpretation

The relatively high density indicates that respondents' answers to many different statements tend to vary together. This suggests substantial shared structure in the class's response patterns rather than completely independent opinions.

---

# 3. Sign of the Correlations

The final network contains:

- Positive edges: **810**
- Negative edges: **0**

Every retained edge therefore has

$$
r_{ij} \geq 0.30.
$$

A positive correlation means that respondents who give relatively high responses to one statement also tend to give relatively high responses to the other statement, and respondents who give relatively low responses to one tend to give relatively low responses to the other.

### Interpretation

All sufficiently strong relationships retained in the network are positive. Thus, among the relationships that satisfy the selected threshold, respondents' opinions generally move in the **same direction** across statement pairs.

This does **not** mean that the class has no disagreements or opposing opinions. It only means that no negative correlation reached the threshold

$$
|r| \geq 0.30.
$$

Weaker negative correlations may still exist in the full correlation matrix but are not represented as edges in the final thresholded network.

---

# 4. Connected Components

The network contains **4 connected components**.

The largest component contains:

$$
57/60 = 95\%
$$

of all statements.

The remaining three statements are isolated:

- **E02:** Traditional written examinations accurately measure a student's knowledge.
- **E03:** Class attendance should be compulsory for all courses.
- **E04:** High-quality online learning can effectively complement classroom teaching.

An isolated statement has no other statement with

$$
|r| \geq 0.30.
$$

### Interpretation

E02, E03, and E04 do not have sufficiently strong correlations with any other statement to appear in the thresholded network.

The fact that 57 out of 60 statements belong to one large connected component indicates that the survey responses are dominated by a single broad interconnected opinion structure rather than several equally sized disconnected groups.

---

# 5. Degree Centrality

## Definition

The degree of a statement is simply the number of other statements to which it is connected.

For a node \(i\),

$$
\text{Degree}(i)
=
\text{number of edges incident to } i.
$$

Since an edge exists only when

$$
|r_{ij}| \geq 0.30,
$$

degree measures how many other statements have a sufficiently strong correlation with that statement.

## Results

| Statement | Category | Degree |
|---|---|---:|
| S09 | S | 45 |
| S15 | S | 44 |
| S07 | S | 43 |
| E11 | E | 42 |
| E14 | E | 42 |

The most connected statement is **S09**, with degree 45.

S09 is:

> "Universities should promote an inclusive environment where different viewpoints can be discussed respectfully."

It is connected to 45 of the other 59 statements.

### Interpretation

S09 has the broadest set of sufficiently strong associations in the network. This indicates that the response pattern for the issue of inclusive university environments is associated with a relatively large number of other opinions.

S15 also has a high degree of 44:

> "Building public trust is essential for responsible adoption of emerging technologies."

This indicates that responses to the issue of public trust are also strongly associated with many other statements.

### Conclusion from degree

The degree results identify statements with **broad structural connectivity** within the opinion network.

S09, S15, S07, E11, and E14 are particularly well connected.

---


# 6. Eigenvector Centrality

## Definition

Eigenvector centrality considers not only how many connections a node has, but also how central its neighbors are.

The basic idea is:

> A statement is structurally important when it is connected to other structurally important statements.

For the weighted network, the analysis uses the magnitude of the correlations:

$$
w_{ij} = |r_{ij}|.
$$

This allows both positive and negative correlations to be treated as strengths when calculating eigenvector centrality.

## Results

| Statement | Category | Eigenvector Centrality |
|---|---|---:|
| S09 | S | 0.1993 |
| V07 | V | 0.1907 |
| V14 | V | 0.1845 |
| S15 | S | 0.1842 |
| V15 | V | 0.1814 |

S09 has the highest eigenvector centrality.

Although degree already shows that S09 is highly connected, eigenvector centrality provides additional information: S09 is connected to statements that are themselves highly connected.

V07 and V14 are environmental statements and also have high eigenvector centrality, indicating that environmental issues occupy central positions in the overall interconnected opinion structure.

### Conclusion from eigenvector centrality

The eigenvector-centrality results suggest that S09, V07, V14, S15, and V15 are embedded within the **core of the network's highly interconnected structure**.

The top statements are drawn from more than one survey category, suggesting that structurally central opinions are not confined to a single predefined T/E/S/V category.

---

# 7. Betweenness Centrality

## Definition

Betweenness centrality measures how often a node lies on shortest paths connecting other pairs of nodes.

For this analysis, edge strength is converted into distance using

$$
d_{ij} = 1 - |r_{ij}|.
$$

Therefore:

- \( |r| -> 1 \) gives a very short distance.
- \( |r| -> 0 \) gives a longer distance.

For example:

$$
|r|=0.90
\quad\Rightarrow\quad
d=0.10
$$

while

$$
|r|=0.40
\quad\Rightarrow\quad
d=0.60.
$$

This makes strongly correlated statements closer together in the graph.

## Results

| Statement | Category | Betweenness Centrality |
|---|---|---:|
| S15 | S | 0.0479 |
| V09 | V | 0.0421 |
| T07 | T | 0.0374 |
| S07 | S | 0.0368 |
| S13 | S | 0.0362 |

S15 has the highest betweenness centrality.

S15 is:

> "Building public trust is essential for responsible adoption of emerging technologies."

### Interpretation

A high betweenness value means that a statement frequently lies on relatively short paths connecting other statements.

Therefore, S15 occupies a potential **bridging position** between different parts of the opinion network.

V09 and T07 also have relatively high betweenness, suggesting that they may connect parts of the network even though they do not necessarily have the very highest degree.

For example, T07:

> "Robots should replace humans in hazardous occupations whenever possible."

does not appear among the top five degree-centrality statements, but it has the third-highest betweenness.

This shows that a statement can have a relatively strong bridging role without having the largest number of direct connections.

### Conclusion from betweenness

S15 appears particularly important from a **network-bridging** perspective. It may connect otherwise less directly connected sets of statements.

However, betweenness does not imply causal influence. It only describes the statement's structural position in the correlation network.

---

# 8. Comparison of Centrality Measures

The three centrality measures capture different properties of the network.

| Metric | What it measures | Interpretation |
|---|---|---|
| Degree | Number of strong connections | Broad connectivity |
| Eigenvector centrality | Connections to other central nodes | Position within the network core |
| Betweenness centrality | Frequency of lying on shortest paths | Bridging/connectivity between network regions |

Several statements appear prominently across multiple measures.

### S09

S09 has:

- Highest degree
- Highest eigenvector centrality

This indicates both broad connectivity and connections to other highly central statements.

### S15

S15 has:

- Very high degree
- High eigenvector centrality
- Highest betweenness

This makes S15 particularly prominent both as a highly connected statement and as a potential bridge between other parts of the network.

### T07

T07 illustrates the difference between degree and betweenness. It is not among the most connected statements, but it has high betweenness, suggesting that its importance comes more from its position between other nodes than from its number of direct links.

---

# 9. Relationship with Survey Categories

The survey was originally divided into T/E/S/V categories.

The highest-degree statements are:

- S09 — S
- S15 — S
- S07 — S
- E11 — E
- E14 — E

The highest-eigenvector-centrality statements include:

- S09 — S
- V07 — V
- V14 — V
- S15 — S
- V15 — V

The highest-betweenness statements include:

- S15 — S
- V09 — V
- T07 — T
- S07 — S
- S13 — S

Thus, high-centrality statements occur across several categories rather than belonging exclusively to a single category.

### Interpretation

This suggests that the observed structure of respondents' answers is not organized entirely according to the questionnaire's predefined thematic categories.

For example, environmental statements such as V07, V14, and V15 appear alongside social statements such as S09 and S15 among the most central nodes.

Therefore, the response network reveals relationships across the survey's original thematic boundaries.

This should be interpreted as evidence of **cross-category association in the response patterns**, rather than as evidence that the original categories are incorrect.

---

# 10. Threshold Selection and Robustness

Different thresholds produced the following networks:

| Threshold | Edges | Density | Negative Edges |
|---:|---:|---:|---:|
| 0.10 | 1546 | 0.8734 | 31 |
| 0.20 | 1211 | 0.6842 | 5 |
| 0.25 | 1021 | 0.5768 | 3 |
| 0.30 | 810 | 0.4576 | 0 |

At 0.10 the graph is extremely dense, with 87.3% of all possible edges retained. Such a network provides relatively little differentiation between nodes because almost every statement is connected to almost every other statement.

Increasing the threshold progressively removes weaker relationships and produces a more selective structure.

The 0.30 network retains 810 edges while preserving one large connected component containing 57 of the 60 statements. It was therefore selected as the final network.

An additional observation is that several statements remain prominent across different thresholds. In particular, S09, S15, V07, V14, and V15 repeatedly appear among the highly central statements.

This stability suggests that their structural prominence is not entirely dependent on a single threshold choice.

However, the network results should still be interpreted as being conditional on the selected threshold.

---

# 11. Overall Conclusions

The statement-statement opinion network reveals a strongly interconnected structure in the class's response patterns.

Using a Spearman correlation threshold of

$$
|r| \geq 0.30,
$$

the network contains 60 statements and 810 edges out of 1,770 possible statement pairs, giving a density of 0.4576. The average statement is connected to 27 other statements.

The network is dominated by one large connected component containing 57 of the 60 statements. E02, E03, and E04 are isolated because none of their correlations with other statements reaches the selected threshold.

All retained edges are positive. Therefore, among the sufficiently strong relationships represented in the network, respondents' answers to statement pairs generally move in the same direction. No negative correlation reaches the threshold of 0.30, although weaker negative correlations may still exist in the full correlation matrix.

The centrality analysis identifies several structurally prominent statements.

**S09** has the highest degree and highest eigenvector centrality, indicating that it has broad connections and is connected to other highly connected statements.

**S15** has high degree, high eigenvector centrality, and the highest betweenness centrality, indicating that the public-trust statement is both broadly connected and structurally positioned as a bridge between other statements.

Other highly central statements include V07, V14, V15, S07, E11, E14, V09, T07, and S13.

The high-centrality statements span multiple survey categories, particularly S, V, E, and T. This indicates that the strongest structural relationships in the response data are not confined to the survey's predefined thematic categories.

Overall, the network suggests that the class's opinions exhibit substantial **cross-statement and cross-category structure**, with some issues occupying central or bridging positions within this structure.

The analysis describes **associations between response patterns**, not absolute agreement, importance, or causality. In particular, a highly connected statement is not necessarily one with high average agreement, and a high-betweenness statement does not necessarily cause or influence opinions on other issues.