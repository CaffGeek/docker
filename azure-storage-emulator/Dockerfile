FROM microsoft/windowsservercore

# SQL Server 2012 LocalDB
COPY SqlLocalDB.msi .

RUN powershell Start-Process -FilePath msiexec -ArgumentList /q, /i, SqlLocalDB.msi, IACCEPTSQLLOCALDBLICENSETERMS=YES -Wait;
WORKDIR 'C:\\Program Files\\Microsoft SQL Server\\110\\Tools\\Binn\\'
RUN powershell SqlLocalDB.exe create azure -s

# Azure Storage Emulator
ADD  https://go.microsoft.com/fwlink/?linkid=717179&clcid=0x409 MicrosoftAzureStorageEmulator.msi
RUN powershell Start-Process -FilePath msiexec -ArgumentList /q, /i, MicrosoftAzureStorageEmulator.msi -Wait;
WORKDIR 'C:\\Program Files (x86)\\Microsoft SDKs\\Azure\\Storage Emulator\\'
RUN AzureStorageEmulator.exe init /server '(localdb)\azure'
EXPOSE 10000 10001 10002

# Configure and launch
COPY start.ps1 .
CMD powershell .\start