from ftplib import FTP

with FTP("ftp.wsdot.wa.gov") as ftp:

    ftp.login()
    ftp.cwd('/public/Environmental_SC/Species_Videos/HD_CALA/')

    ftp.dir()
