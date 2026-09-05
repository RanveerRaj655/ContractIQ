# Regression Test Results (Guarded Pipeline)

## Question 1 (Answerable)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: False (Score: -2.0648)
**Hallucination Verdict**: UNKNOWN

**Query**: Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. Details: Which state/country's law governs the interpretation of the contract?

| Expected | Actual |
|----------|--------|
| Requires extraction of governing law (usually New York, Delaware, etc.). | The governing law for the contracts provided is as follows:<br><br>1. **CUROGROUP HOLDINGS CORP**: The agreement is governed by the laws of the State of New York [Chunk 1].<br>2. **BELLICUM PHARMACEUTICALS INC**: This agreement is also governed by the laws of the State of New York, USA [Chunk 2].<br>3. **Verizon Abs LLC**: The agreement is governed by the internal laws of the State of New York [Chunk 3].<br>4. **Gentech Holdings Inc**: This agreement is governed by the laws of the State of Florida [Chunk 4].<br>5. **Reynolds Consumer Products Inc**: The agreement is governed by the laws of the State of Illinois and the United States of America [Chunk 5].<br><br>A lawyer should review the specific governing law clauses in each contract to ensure compliance and understanding of the applicable laws in New York, Florida, and Illinois. |

## Question 2 (Answerable)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: False (Score: -1.1718)
**Hallucination Verdict**: UNKNOWN

**Query**: Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective

| Expected | Actual |
|----------|--------|
| Requires extraction of the effective date. | The relevant parts of the contracts related to "Effective Date" that should be reviewed by a lawyer are as follows:<br><br>1. In Chunk 1, the "Effective Date" is defined as "the date first written above."<br>2. In Chunk 2, the term of the agreement commences on the "Effective Date."<br>3. In Chunk 3, the "Term" of the agreement also commences on the "Effective Date."<br>4. In Chunk 4, the agreement is effective on the date of signed acceptance, referred to as the "Effective Date."<br>5. In Chunk 5, the agreement becomes effective on the date first written above.<br><br>These definitions and conditions regarding the "Effective Date" are crucial for understanding when the obligations and rights under the contracts commence and may require legal scrutiny to ensure clarity and enforceability. |

## Question 3 (Answerable)
**Target Document**: `PfHospitalityGroupInc_20150923_10-12G_EX-10_1_9266710_EX-10_1_Franchise_Agreement3.txt`
**Abstained**: False (Score: -1.7418)
**Hallucination Verdict**: SUPPORTED

**Query**: Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract

| Expected | Actual |
|----------|--------|
| Requires extraction of the contracting parties. | I don't have enough information in the provided contracts to answer this. |

## Question 4 (Answerable)
**Target Document**: `PfHospitalityGroupInc_20150923_10-12G_EX-10_1_9266710_EX-10_1_Franchise_Agreement3.txt`
**Abstained**: False (Score: -3.5350)
**Hallucination Verdict**: SUPPORTED

**Query**: Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract

| Expected | Actual |
|----------|--------|
| Requires extraction of the agreement date. | I don't have enough information in the provided contracts to answer this. |

## Question 5 (Answerable)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: False (Score: -3.2274)
**Hallucination Verdict**: SUPPORTED

**Query**: Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?

| Expected | Actual |
|----------|--------|
| Requires details about termination for convenience. | I don't have enough information in the provided contracts to answer this. |

## Question 6 (Answerable)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: True (Score: -4.2361)
**Hallucination Verdict**: N/A

**Query**: Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing" that should be reviewed by a lawyer. Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?

| Expected | Actual |
|----------|--------|
| Requires details on revenue or profit sharing. | Insufficient context (Score -4.2361 < Threshold -4.0). Abstained. |

## Question 7 (Adversarial)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: True (Score: -5.3373)
**Hallucination Verdict**: N/A

**Query**: Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty?

| Expected | Actual |
|----------|--------|
| I don't have enough information in the provided contracts to answer this. | Insufficient context (Score -5.3373 < Threshold -4.0). Abstained. |

## Question 8 (Adversarial)
**Target Document**: `DOMINIADVISORTRUST_02_18_2005-EX-99__H__2_-SPONSORSHIP_AGREEMENT.txt`
**Abstained**: False (Score: -0.3882)
**Hallucination Verdict**: UNKNOWN

**Query**: Highlight the parts (if any) of this contract related to "Force Majeure" that should be reviewed by a lawyer.

| Expected | Actual |
|----------|--------|
| I don't have enough information in the provided contracts to answer this. | The following parts related to "Force Majeure" from the provided contracts should be reviewed by a lawyer:<br><br>1. **Definition and Scope**: Each contract defines what constitutes a "Force Majeure" event, including examples such as acts of God, war, and government actions. The specific language and examples used may have legal implications and should be carefully considered.<br><br>2. **Liability and Breach**: The contracts state that neither party will be liable for delays or failures to perform due to Force Majeure events. The conditions under which a party is excused from performance and the implications of such provisions should be analyzed.<br><br>3. **Notice Requirements**: The requirement for the affected party to give prompt written notice of a Force Majeure event is a critical aspect. The specifics of how and when notice must be given can affect the rights and obligations of the parties.<br><br>4. **Duration and Cure**: The provisions regarding the duration of the Force Majeure event and the obligation to take reasonable steps to cure the cause of the delay are important. The language surrounding these obligations can impact the enforceability of the contract.<br><br>5. **Termination Rights**: In some contracts, there are stipulations regarding the right to terminate the agreement if the delay exceeds a certain period (e.g., 60 days). This aspect should be reviewed to understand the potential consequences of prolonged Force Majeure events.<br><br>These elements are crucial for understanding the legal ramifications of Force Majeure clauses and should be carefully examined by a legal professional. |

## Question 9 (Adversarial)
**Target Document**: `PfHospitalityGroupInc_20150923_10-12G_EX-10_1_9266710_EX-10_1_Franchise_Agreement3.txt`
**Abstained**: False (Score: -3.3353)
**Hallucination Verdict**: SUPPORTED

**Query**: Highlight the parts (if any) of this contract related to "Cap On Liability" that should be reviewed by a lawyer. Details: Does the contract include a cap on liability upon the breach of a party’s obligation?

| Expected | Actual |
|----------|--------|
| I don't have enough information in the provided contracts to answer this. | I don't have enough information in the provided contracts to answer this. |

## Question 10 (Adversarial)
**Target Document**: `PfHospitalityGroupInc_20150923_10-12G_EX-10_1_9266710_EX-10_1_Franchise_Agreement3.txt`
**Abstained**: True (Score: -5.9132)
**Hallucination Verdict**: N/A

**Query**: Highlight the parts (if any) of this contract related to "Exclusivity" that should be reviewed by a lawyer. Details: Is one party granted an exclusive right?

| Expected | Actual |
|----------|--------|
| I don't have enough information in the provided contracts to answer this. | Insufficient context (Score -5.9132 < Threshold -4.0). Abstained. |

