<!DOCTYPE html>
<html lang="en">
<head>
    <title>Legacy report</title>
</head>
<body>
<?php

require __DIR__ . '/../bootstrap/legacy.php';

function formatRow(array $row): string
{
    return sprintf('%s — %d', $row['label'], $row['count']);
}

$rows = fetch_report_rows();

?>
<table>
    <?php foreach ($rows as $row): ?>
        <tr><td><?= htmlspecialchars(formatRow($row)) ?></td></tr>
    <?php endforeach; ?>
</table>
</body>
</html>
