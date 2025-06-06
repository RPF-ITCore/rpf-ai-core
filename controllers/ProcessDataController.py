from .BaseController import BaseController
from .ProjectController import ProjectController
from enums import FileEnum
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import logging

class ProcessDataController(BaseController):
    def __init__(self, project_id : str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=self.project_id)
        self.logger = logging.getLogger(__name__)

    
    def get_file_extension(self, file_id : str):
        return os.path.splitext(file_id)[-1]
    
    def get_file_loader(self , file_id : str):

        file_ext = self.get_file_extension(file_id)
        #self.logger.info(f"File Extension: {file_ext}")
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if not os.path.exists(file_path):
            self.logger.error(f"file doesn't exists, NOT FOUND, {file_path}") # file doesn't exists, NOT FOUND 
            return None

        if file_ext == FileEnum.PDF.value:
            return PyMuPDFLoader(file_path=file_path)
        if file_ext == FileEnum.TXT.value:
            return TextLoader(file_path=file_path)
        
        return None


    def get_file_content(self, file_id : str):

        loader = self.get_file_loader(file_id=file_id)

        if loader:
            return loader.load()
        
        else:
            self.logger.error(f"No file loader found for file_id: {file_id}")
            return None
        
        
    

    def process_file_content(self, file_content : list, chunk_size : int = 1200, overlap_size : int = 20):

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                length_function=len
            )
            file_content_texts = [
                rec.page_content
                for rec in file_content
            ]

            file_content_metadata = [
                rec.metadata
                for rec in file_content
            ]

            chunks = text_splitter.create_documents(file_content_texts , metadatas = file_content_metadata)
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error processing file content: {e}")
            raise e
        
    
