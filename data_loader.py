"""
Data ingestion module for the Legal Judgment Summarization Pipeline.

Provides three loading strategies:
  1. load_ildc()           — HuggingFace ILDC dataset
  2. load_from_directory() — local .txt files
  3. load_sample_judgment()— built-in Indian SC judgment for zero-setup demos

All loaders return a uniform List[Document] for downstream consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Data Model
# ────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single legal document with optional reference summary."""

    doc_id: str
    text: str
    reference_summary: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def approx_token_count(self) -> int:
        """Rough whitespace-based token estimate."""
        return len(self.text.split())


# ────────────────────────────────────────────────────────────
# Loader: HuggingFace ILDC
# ────────────────────────────────────────────────────────────

def load_ildc(split: str = "test", max_docs: Optional[int] = None) -> list[Document]:
    """
    Load judgments from the ILDC (Indian Legal Documents Corpus)
    via the HuggingFace datasets library.

    Args:
        split: Dataset split to load ('train', 'validation', 'test').
        max_docs: Maximum number of documents to load (None = all).

    Returns:
        List of Document objects.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for ILDC loading. "
            "Install it with: pip install datasets"
        )

    logger.info("Loading ILDC dataset (split=%s) from HuggingFace…", split)
    ds = load_dataset("coastalcph/fairlex_ildc", split=split)

    documents: list[Document] = []
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        documents.append(
            Document(
                doc_id=f"ildc_{split}_{i}",
                text=row.get("text", ""),
                reference_summary=row.get("summary", None),
                metadata={"source": "ildc", "split": split, "label": row.get("label", None)},
            )
        )

    logger.info("Loaded %d documents from ILDC.", len(documents))
    return documents


# ────────────────────────────────────────────────────────────
# Loader: Local Directory
# ────────────────────────────────────────────────────────────

def load_from_directory(
    directory: str | Path,
    glob_pattern: str = "*.txt",
    reference_dir: Optional[str | Path] = None,
) -> list[Document]:
    """
    Load judgment texts from a local directory.

    If *reference_dir* is provided, it is searched for files with matching
    stems (e.g. ``judgment_001.txt`` ↔ ``judgment_001_summary.txt``) to
    populate reference summaries.

    Args:
        directory: Path to the directory containing judgment text files.
        glob_pattern: Glob pattern for matching files.
        reference_dir: Optional directory containing reference summary files.

    Returns:
        List of Document objects.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    ref_map: dict[str, str] = {}
    if reference_dir is not None:
        ref_path = Path(reference_dir)
        if ref_path.is_dir():
            for f in ref_path.glob(glob_pattern):
                ref_map[f.stem.replace("_summary", "")] = f.read_text(
                    encoding="utf-8"
                )

    documents: list[Document] = []
    for filepath in sorted(directory.glob(glob_pattern)):
        text = filepath.read_text(encoding="utf-8")
        ref = ref_map.get(filepath.stem, None)
        documents.append(
            Document(
                doc_id=filepath.stem,
                text=text,
                reference_summary=ref,
                metadata={"source": "local", "path": str(filepath)},
            )
        )

    logger.info("Loaded %d documents from %s.", len(documents), directory)
    return documents


# ────────────────────────────────────────────────────────────
# Loader: Built-in Sample Judgment
# ────────────────────────────────────────────────────────────

_SAMPLE_JUDGMENT = """
SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION
CIVIL APPEAL NO. 5765 OF 2019

M/s. Kiran Constructions Pvt. Ltd.          ... Appellant(s)
                     VERSUS
State of Maharashtra & Ors.                 ... Respondent(s)

JUDGMENT

Hon'ble Mr. Justice R.K. Agrawal
Hon'ble Ms. Justice Indira Banerjee

Date of Judgment: 15th March, 2019

1. This appeal arises out of the judgment and order dated 12th September, 2018 passed by the High Court of Judicature at Bombay in Writ Petition No. 3421 of 2017, whereby the Division Bench of the High Court dismissed the writ petition filed by the appellant challenging the acquisition of land bearing Survey No. 145/2 situated in Village Khed, Taluka Mulshi, District Pune.

2. The brief facts of the case are as follows: The appellant is a private limited company engaged in the business of construction and real estate development. The appellant is the owner of land admeasuring 5 acres and 20 guntas bearing Survey No. 145/2 situated in Village Khed, Taluka Mulshi, District Pune (hereinafter referred to as "the subject land"). The appellant purchased the subject land through a registered sale deed dated 5th January, 2010 from the original owner, Shri Ramchandra Patil.

3. On 22nd August, 2015, the Government of Maharashtra issued a notification under Section 11(1) of the Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act, 2013 (hereinafter referred to as "the Act of 2013") for the acquisition of the subject land for the purpose of construction of a bypass road connecting the Pune-Mumbai Expressway to National Highway 48.

4. The appellant filed objections under Section 15 of the Act of 2013 on the following grounds: (a) The acquisition was not for a public purpose as the proposed bypass road would primarily benefit a private industrial estate located nearby; (b) Alternative land was available for the construction of the bypass road which would cause less displacement; (c) The Social Impact Assessment (SIA) study was not conducted in accordance with the mandatory provisions of Sections 4 to 8 of the Act of 2013; (d) The compensation offered was grossly inadequate and did not reflect the true market value of the land.

5. The District Collector, after considering the objections, rejected the same by order dated 3rd February, 2016 and recommended the acquisition to proceed. The appellant thereafter filed a writ petition before the High Court of Judicature at Bombay challenging the acquisition proceedings.

6. The High Court, by the impugned judgment, dismissed the writ petition holding that: (a) The acquisition was for a bonafide public purpose, namely the construction of a bypass road; (b) The SIA study was substantially compliant with the provisions of the Act of 2013; (c) The adequacy of compensation could not be examined in writ proceedings under Article 226 of the Constitution of India and the appellant had an alternative remedy before the Land Acquisition, Rehabilitation and Resettlement Authority under Section 64 of the Act of 2013.

7. Learned Senior Counsel appearing for the appellant submitted that the High Court erred in holding that the SIA study was substantially compliant with the Act of 2013. He contended that Section 4(1) of the Act mandates that the appropriate Government shall ensure that a Social Impact Assessment study is carried out in the affected area in such manner as may be prescribed. He further submitted that Section 4(2) requires the study to include the following: (a) assessment of the nature and extent of displacement; (b) an estimate of the number of families likely to be affected; (c) the extent of lands, public and private, houses, settlements and other common properties likely to be affected by the proposed acquisition; (d) whether the extent of land proposed is the minimum necessary for the project; (e) alternative sites, if any, which have been considered and a specific statement on why they are not suitable.

8. The appellant's counsel further argued that in the present case, the SIA report was prepared without conducting any public hearing as mandated by Section 5 of the Act of 2013, which requires the appropriate Government to notify the affected families and seek their views. He placed reliance on the decision of this Court in Pune Municipal Corporation v. Harakchand Misirimal Solanki, (2014) 3 SCC 183, wherein it was held that compliance with SIA provisions is mandatory and not directory.

9. Per contra, the learned Additional Solicitor General appearing for the State submitted that the SIA study was conducted in substantial compliance with the Act of 2013. He further submitted that minor procedural irregularities in the SIA process do not vitiate the acquisition proceedings. Reliance was placed on the decision of this Court in Radhy Shyam v. State of U.P., (2011) 5 SCC 553.

10. We have heard learned counsel for both parties at length and have carefully perused the record.

11. The primary question that arises for our consideration is whether the Social Impact Assessment study conducted in the present case complied with the mandatory requirements of Sections 4 and 5 of the Act of 2013.

12. It is well settled that the Act of 2013 was enacted to replace the archaic Land Acquisition Act, 1894 with the specific objective of ensuring transparency, fair compensation, and rehabilitation and resettlement of affected persons. The requirement of SIA study under Sections 4 to 8 of the Act of 2013 is a significant safeguard introduced by the legislature to protect the interests of landowners and affected families.

13. Upon careful examination of the SIA report placed on record, we find that: (a) The SIA report does not contain any assessment of alternative sites as mandated by Section 4(2)(e) of the Act of 2013. The report merely states that "no alternative site is feasible" without providing any analysis or reasons therefor; (b) No public hearing was conducted as required by Section 5 of the Act of 2013. The State has not produced any evidence of notification being issued to affected families for seeking their views; (c) The SIA report does not contain any assessment of the minimum extent of land required for the project as mandated by Section 4(2)(d).

14. In light of the above findings, we are of the considered opinion that the SIA study conducted in the present case does not comply with the mandatory requirements of Sections 4 and 5 of the Act of 2013. The decision of this Court in Pune Municipal Corporation v. Harakchand Misirimal Solanki (supra) clearly holds that the provisions of Sections 4 to 8 are mandatory in nature and non-compliance therewith vitiates the acquisition proceedings.

15. We are unable to accept the submission of the learned Additional Solicitor General that the irregularities were minor and did not prejudice the appellant. The failure to assess alternative sites, the absence of public hearing, and the failure to determine the minimum extent of land required are not mere procedural irregularities but go to the root of the purpose for which the SIA study is mandated under the Act.

16. Regarding the question of compensation, we agree with the High Court that the adequacy of compensation cannot be examined in these proceedings. The appellant is at liberty to raise this issue before the Land Acquisition, Rehabilitation and Resettlement Authority under Section 64 of the Act of 2013.

17. In the result, this appeal is allowed in part. The judgment and order of the High Court dated 12th September, 2018 and the notification under Section 11(1) of the Act of 2013 dated 22nd August, 2015 are hereby set aside to the extent that they relate to the acquisition of the subject land. The matter is remanded to the State Government with a direction to conduct a fresh Social Impact Assessment study in strict compliance with Sections 4 and 5 of the Act of 2013. The State Government shall complete the fresh SIA study within a period of six months from the date of this judgment.

18. However, we make it clear that this judgment shall not preclude the State Government from proceeding with the acquisition afresh in accordance with law after completing the SIA study in compliance with the provisions of the Act of 2013.

19. There shall be no order as to costs.

20. Pending applications, if any, are disposed of.

                                        ……………………………………J.
                                        (R.K. AGRAWAL)

                                        ……………………………………J.
                                        (INDIRA BANERJEE)
New Delhi;
March 15, 2019.
""".strip()

_SAMPLE_REFERENCE = (
    "The Supreme Court in M/s. Kiran Constructions Pvt. Ltd. v. State of Maharashtra "
    "allowed the appeal in part, setting aside the High Court's dismissal of the writ "
    "petition challenging land acquisition. The Court found that the Social Impact "
    "Assessment (SIA) study conducted under the Right to Fair Compensation and "
    "Transparency in Land Acquisition, Rehabilitation and Resettlement Act, 2013 did "
    "not comply with the mandatory requirements of Sections 4 and 5. Specifically, "
    "the SIA report failed to assess alternative sites (Section 4(2)(e)), no public "
    "hearing was conducted (Section 5), and no assessment of minimum land extent was "
    "made (Section 4(2)(d)). The Court remanded the matter to the State Government "
    "with a direction to complete a fresh SIA study within six months. The Court "
    "upheld the High Court's finding that adequacy of compensation could not be "
    "examined in writ proceedings and directed the appellant to the LARR Authority "
    "under Section 64 of the Act."
)


def load_sample_judgment() -> list[Document]:
    """
    Return a built-in sample Indian Supreme Court judgment for instant
    zero-setup testing of the pipeline.

    Returns:
        A single-element list containing the sample Document.
    """
    doc = Document(
        doc_id="sample_sc_judgment_001",
        text=_SAMPLE_JUDGMENT,
        reference_summary=_SAMPLE_REFERENCE,
        metadata={"source": "builtin_sample", "court": "Supreme Court of India"},
    )
    logger.info(
        "Loaded built-in sample judgment (%d chars, ~%d tokens).",
        doc.char_count,
        doc.approx_token_count,
    )
    return [doc]
