<?php

declare(strict_types=1);

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

interface Moderatable
{
    public function isApproved(): bool;
}

trait RecordsModeration
{
    public function moderationTrail(): array
    {
        return [
            'approved_at' => $this->approved_at,
            'rejected_at' => $this->rejected_at,
        ];
    }
}

enum ReviewStatus: string
{
    case Pending = 'pending';
    case Approved = 'approved';
    case Rejected = 'rejected';
}

/**
 * A customer review left against a product.
 */
class Review extends Model implements Moderatable
{
    use RecordsModeration;

    protected $fillable = ['body', 'rating', 'product_id'];

    protected $casts = [
        'approved_at' => 'datetime',
        'rejected_at' => 'datetime',
    ];

    public function product(): BelongsTo
    {
        return $this->belongsTo(Product::class);
    }

    public function isApproved(): bool
    {
        return $this->approved_at !== null && $this->rejected_at === null;
    }

    public function status(): ReviewStatus
    {
        if ($this->rejected_at !== null) {
            return ReviewStatus::Rejected;
        }

        return $this->approved_at !== null
            ? ReviewStatus::Approved
            : ReviewStatus::Pending;
    }
}

function normalizeRating(int|float $raw): int
{
    return (int) max(1, min(5, round($raw)));
}
