docker container run -p 20000:10000 -p 20001:10001 -p 20002:10002 --name azure-storage-emulator -d azure-storage-emulator

Then connect with the connection string

DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:20000/devstoreaccount1;TableEndpoint=http://127.0.0.1:20002/devstoreaccount1;QueueEndpoint=http://127.0.0.1:20001/devstoreaccount1

ISSUES:
Currently, the latest Storage Explorer won't let you connect because you need a SAS token.