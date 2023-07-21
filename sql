DECLARE @BatchSize INT = 1000; -- Set your desired batch size

DECLARE @SDFKey NVARCHAR(MAX);

DECLARE db_cursor CURSOR FOR
SELECT DISTINCT Source_Data_File_Reference_Key
FROM DANDT..RLEN_PLUS_FINAL WITH (NOLOCK)
ORDER BY Source_Data_File_Reference_Key DESC;

OPEN db_cursor;

FETCH NEXT FROM db_cursor INTO @SDFKey;

WHILE @@FETCH_STATUS = 0
BEGIN
    PRINT CONCAT('Begin:', @SDFKey);

    BEGIN TRAN;

    INSERT INTO DNT_HIST..RLEN_PLUS_FINAL WITH (TABLOCK) -- Use TABLOCK hint
    SELECT TOP (@BatchSize) *
    FROM DANDT..RLEN_PLUS_FINAL WITH (NOLOCK)
    WHERE Source_Data_File_Reference_Key = @SDFKey;

    COMMIT;

    PRINT CONCAT('FINISH:', @SDFKey);

    FETCH NEXT FROM db_cursor INTO @SDFKey;
END

CLOSE db_cursor;
DEALLOCATE db_cursor;
