import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

from core.config import get_settings
from knowledgebase_data.json_data_processor import json_to_bilingual_text
from llm.providers.OpenAIProvider import OpenAIProvider
from vectordb.VectorDBEnums import VectorDBEnums
from vectordb.VectorDBProviderFactory import VectorDBProviderFactory


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


class JobarEmbeddingPipeline:
    def __init__(
        self,
        json_path: Path,
        text_output_path: Path,
        metadata_output_path: Path,
        collection_name: str,
        batch_size: int = 16,
        batch_delay: float = 0.0,
        reset_collection: bool = False,
        upload_batch_size: int = 10,
        qdrant_timeout: float = 120.0,
    ):
        self.json_path = json_path
        self.text_output_path = text_output_path
        self.metadata_output_path = metadata_output_path
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.reset_collection = reset_collection
        self.upload_batch_size = upload_batch_size
        self.qdrant_timeout = qdrant_timeout

        self.settings = get_settings()

        self.embedding_model_id = (
            self.settings.EMBEDDING_MODEL_ID
            if self.settings.EMBEDDING_MODEL_ID
            else "text-embedding-3-large"
        )
        try:
            self.embedding_size = int(self.settings.EMBEDDING_MODEL_SIZE)
        except (TypeError, ValueError):
            # Fall back to the expected dimension for text-embedding-3-large
            self.embedding_size = 3072

        self.openai_client = OpenAIProvider(
            api_key=self.settings.OPENAI_API_KEY,
            api_url=self.settings.OPENAI_API_URL,
            default_input_max_characters=self.settings.DEFAULT_INPUT_MAX_CHARACTERS or 8000,
            default_generation_max_output_tokens=self.settings.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS or 1024,
            default_generation_temperature=self.settings.DEFAULT_GENERATION_TEMPREATUER or 0.0,
        )
        self.openai_client.set_embedding_model(
            model_id=self.embedding_model_id,
            embedding_size=self.embedding_size,
        )

        self.qdrant_provider = self._create_qdrant_provider()
        self.records: List[Dict[str, str]] = []
        self.embeddings: List[Sequence[float]] = []

    def _create_qdrant_provider(self):
        factory = VectorDBProviderFactory(self.settings)
        provider = factory.create(VectorDBEnums.QDRANT.value)
        if not provider:
            raise RuntimeError("Failed to instantiate Qdrant provider. Check VECTORDB_BACKEND setting.")

        provider.connect(
            url=self.settings.VECTORDB_URL,
            api_key=self.settings.QDRANT_API_KEY,
            timeout=self.qdrant_timeout,
        )
        return provider

    def _convert_to_text_blocks(self):
        logger.info("Converting JSON rows into bilingual text blocks…")
        logger.info("Each JSON record will be treated as a separate chunk.")
        self.records = json_to_bilingual_text(
            json_path=str(self.json_path),
            output_path=str(self.text_output_path),
            metadata_output_path=str(self.metadata_output_path),
        )
        logger.info("✅ Prepared %d text blocks (chunks).", len(self.records))
        if self.records:
            logger.info("Sample chunk ID: %s (City: %s, Aspect: %s)", 
                       self.records[0]["id"], 
                       self.records[0]["city"], 
                       self.records[0]["aspect"])

    def _embed_records(self):
        logger.info(
            "Embedding %d text blocks with model %s in batches of %d.",
            len(self.records),
            self.embedding_model_id,
            self.batch_size,
        )
        texts = [record["text"] for record in self.records]
        embeddings: List[Sequence[float]] = []

        for start in range(0, len(texts), self.batch_size):
            end = start + self.batch_size
            batch_texts = texts[start:end]
            logger.info("Embedding batch %d-%d", start, end - 1)
            try:
                batch_vectors = self.openai_client.batch_embed(batch_texts)
            except Exception as exc:
                logger.exception("Embedding batch failed: %s", exc)
                raise

            if not batch_vectors:
                raise RuntimeError(f"Failed to embed records in range {start}-{end}")

            embeddings.extend(batch_vectors)
            if self.batch_delay:
                time.sleep(self.batch_delay)

        if len(embeddings) != len(self.records):
            raise RuntimeError(
                f"Embeddings count {len(embeddings)} does not match records {len(self.records)}"
            )

        self.embeddings = embeddings
        logger.info("✅ Finished embedding %d chunks (records).", len(self.embeddings))
        logger.info("Each chunk has been embedded separately with dimension %d.", self.embedding_size)

    def _prepare_collection(self):
        logger.info(
            "Ensuring Qdrant collection '%s' exists with dimension %d.",
            self.collection_name,
            self.embedding_size,
        )
        created = self.qdrant_provider.create_collection(
            collection_name=self.collection_name,
            embedding_size=self.embedding_size,
            do_reset=self.reset_collection,
        )
        if created:
            logger.info("Created new collection '%s'.", self.collection_name)
        else:
            logger.info("Collection '%s' already present.", self.collection_name)

    def _upsert_records(self):
        logger.info("Uploading embeddings to Qdrant collection '%s'.", self.collection_name)
        metadatas = [
            {
                "block_id": record["id"],
                "sheet": record["sheet"],
                "index": record["index"],
                "city": record["city"],
                "aspect": record["aspect"],
                "subaspect": record["subaspect"],
                "source_json": self.json_path.name,
            }
            for record in self.records
        ]

        texts = [record["text"] for record in self.records]
        record_ids = list(range(len(self.records)))
        success = self.qdrant_provider.insert_many(
            collection_name=self.collection_name,
            texts=texts,
            vectors=self.embeddings,
            metadatas=metadatas,
            record_ids=record_ids,
            batch_size=self.upload_batch_size,
            max_retries=3,
        )

        if not success:
            raise RuntimeError("Failed to insert embeddings into Qdrant.")

        logger.info("✅ Uploaded %d chunks (records) to Qdrant collection '%s'.", 
                   len(self.records), self.collection_name)
        logger.info("Each chunk is stored as a separate document with its own vector and metadata.")

    def _verify_search(self, limit: int = 3):
        """Verify the upload by performing a semantic search with a sample text."""
        if not self.records or not self.embeddings:
            logger.warning("Skipping verification search because no records or embeddings were generated.")
            return

        try:
            # Use the first record's text to create a fresh embedding for verification
            # This ensures we're using the correct model and dimension
            sample_text = self.records[0]["text"]
            logger.info("Running verification search against Qdrant (top %d).", limit)
            logger.info("Re-embedding sample text for verification...")
            
            # Re-embed the sample text to ensure correct dimension
            verification_vector = self.openai_client.embed_text(sample_text)
            if not verification_vector:
                logger.warning("Failed to create verification embedding. Skipping search.")
                return
            
            # Validate dimension matches expected size
            if len(verification_vector) != self.embedding_size:
                logger.error(
                    "Vector dimension mismatch: expected %d, got %d. "
                    "This indicates an embedding model configuration issue.",
                    self.embedding_size,
                    len(verification_vector)
                )
                return
            
            logger.info("Verification vector dimension: %d (expected: %d)", len(verification_vector), self.embedding_size)
            
            results = self.qdrant_provider.search_by_vector(
                collection_name=self.collection_name,
                vector=verification_vector,
                limit=limit,
            )
            if not results:
                logger.warning("Verification search returned no results.")
                return

            logger.info("✅ Verification search successful! Top %d results:", len(results))
            for rank, result in enumerate(results, start=1):
                # Truncate text preview for readability
                text_preview = result.text.replace("\n", " ")[:100]
                logger.info("  %d. Score: %.4f | Preview: %s...", rank, result.score, text_preview)
                
        except Exception as e:
            logger.warning("Verification search failed (non-critical): %s", str(e))
            logger.info("Records were uploaded successfully. Verification search can be skipped.")

    def run(self):
        try:
            self._convert_to_text_blocks()
            if not self.records:
                logger.warning("No records found in %s. Nothing to do.", self.json_path)
                return

            self._embed_records()
            self._prepare_collection()
            self._upsert_records()
            self._verify_search()
        finally:
            if self.qdrant_provider:
                self.qdrant_provider.disconnect()


def parse_args(argv: List[str]) -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    default_json = base_dir / "knowledgebase_data" / "jobar_cleaned_final.json"
    default_text = base_dir / "knowledgebase_data" / "jobar_cleaned_final.txt"
    default_metadata = base_dir / "knowledgebase_data" / "jobar_cleaned_final_blocks.json"

    parser = argparse.ArgumentParser(
        description="Convert JSON data into embeddings and upsert into Qdrant."
    )
    parser.add_argument("--json-path", type=Path, default=default_json, help="Source JSON file.")
    parser.add_argument(
        "--text-output",
        type=Path,
        default=default_text,
        help="Destination .txt file for the bilingual text blocks.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=default_metadata,
        help="Destination .json file for block metadata.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rpf_jobar_embeddings",
        help="Qdrant collection to populate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of records to embed per API call.",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between embedding batches.",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=10,
        help="Number of records to upload per batch to Qdrant (default 10 for cloud instances).",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Qdrant operations (default 120 for cloud instances).",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Drop and recreate the Qdrant collection before inserting.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv or sys.argv[1:])
    pipeline = JobarEmbeddingPipeline(
        json_path=args.json_path,
        text_output_path=args.text_output,
        metadata_output_path=args.metadata_output,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
        reset_collection=args.reset_collection,
        upload_batch_size=args.upload_batch_size,
        qdrant_timeout=args.qdrant_timeout,
    )
    pipeline.run()


if __name__ == "__main__":
    main()


