<?php

declare(strict_types=1);

namespace App\Http\Controllers;

use App\Models\Review;
use App\Models\Shop;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;

/**
 * Read and moderate customer reviews for a shop.
 */
class ReviewController extends Controller
{
    use AuthorizesRequests;

    /** Cache TTL for the review index, in seconds. */
    protected const INDEX_TTL = 300;

    protected Shop $shop;

    public function __construct(Shop $shop)
    {
        $this->shop = $shop;
    }

    /**
     * List reviews for the current shop, newest first.
     *
     * @param  Request  $request  The inbound HTTP request.
     * @return JsonResponse The paginated review collection.
     */
    public function index(Request $request): JsonResponse
    {
        $perPage = (int) $request->query('per_page', '25');

        $reviews = Cache::remember(
            $this->cacheKey($request),
            self::INDEX_TTL,
            function () use ($perPage) {
                return $this->shop
                    ->reviews()
                    ->with(['product', 'author'])
                    ->latest()
                    ->paginate($perPage);
            }
        );

        return response()->json($reviews);
    }

    /**
     * Approve a single review and recompute the product rating.
     */
    public function approve(Request $request, Review $review): JsonResponse
    {
        $this->authorize('moderate', $review);

        $review->forceFill([
            'approved_at' => now(),
            'moderated_by' => $request->user()->id,
        ])->save();

        $review->product->recomputeRating();

        return response()->json(['status' => 'approved']);
    }

    /**
     * Reject a review, recording the moderator's reason.
     */
    public function reject(Request $request, Review $review): JsonResponse
    {
        $this->authorize('moderate', $review);

        $validated = $request->validate([
            'reason' => ['required', 'string', 'max:500'],
        ]);

        $review->forceFill([
            'rejected_at' => now(),
            'rejection_reason' => $validated['reason'],
        ])->save();

        return response()->json(['status' => 'rejected']);
    }

    private function cacheKey(Request $request): string
    {
        return sprintf(
            'shop:%d:reviews:%s',
            $this->shop->id,
            md5($request->fullUrl())
        );
    }
}
